import logging
import math
import re
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)

# Blanket term → (lo_bpm, hi_bpm) core range.
# Calibrated against the actual dataset distribution (n=6397):
#   p10=86, p25=94, p50=120, p75=143, p90=162
_BLANKET_RANGES: Dict[str, Tuple[float, float]] = {
    "very slow":        (40,  75),
    "extremely slow":   (40,  75),
    "slow":             (60,  95),
    "downtempo":        (60,  95),
    "relaxed":          (60, 100),
    "gentle":           (60, 100),
    "mellow":           (60, 100),
    "low-tempo":        (60,  95),
    "medium":           (90, 130),
    "moderate":         (90, 130),
    "mid-tempo":        (90, 130),
    "upbeat":           (115, 150),
    "lively":           (115, 150),
    "brisk":            (120, 150),
    "dance-able":       (115, 150),
    "dance‑able":  (115, 150),  # non-breaking hyphen variant
    "uplifted":         (115, 150),
    "upright":          (120, 160),
    "fast":             (130, 188),
    "high-tempo":       (130, 188),
}

# Terms that carry no tempo information — reward should be skipped.
_SKIP_TERMS = {"none", "free-rubato", "rubato"}


class HardVerifiableRewards():
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def __compute_tempo(self, waveform, sample_rate) -> float:
        import librosa
        import numpy as np
        w = self._to_essentia_array(waveform).astype(np.float32)
        return librosa.beat.tempo(y=w, sr=sample_rate)[0]
    
    
    @staticmethod
    def _to_essentia_array(waveform):
        import numpy as np
        import torch
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.array(waveform, dtype=np.float32)
        if waveform.ndim == 3:
            if waveform.shape[0] != 1:
                raise ValueError(f"Expected a single waveform, got batched audio with shape {waveform.shape}")
            waveform = waveform.squeeze(0)
        if waveform.ndim == 2:          # (channels, samples) or (samples, channels)
            waveform = waveform.mean(axis=0) if waveform.shape[0] <= 8 else waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D mono waveform for Essentia, got shape {waveform.shape}")
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid waveform for Essentia: shape={waveform.shape}")
        return np.ascontiguousarray(waveform)

    def __compute_key(self, waveform, sample_rate) -> dict:
        import essentia.standard as es
        key_extractor = es.KeyExtractor()
        wav_arr = self._to_essentia_array(waveform)
        log.debug("key extraction: shape=%s sr=%s dtype=%s",
                  wav_arr.shape, sample_rate, wav_arr.dtype)
        key, scale, strength = key_extractor(wav_arr)
        return {
            "key": key,
            "scale": scale,
            "strength": strength
        }
    
    
    def extract_attributes(self, waveform, sample_rate) -> Dict[str, float]:
        return {
            "tempo": self.__compute_tempo(waveform, sample_rate),
            "key": self.__compute_key(waveform, sample_rate),
        }
        
    
    # ------------------------------------------------------------------
    # Tempo reward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_target_tempo(expected_tempo: str) -> Optional[Tuple[str, Any]]:
        t = expected_tempo.strip().lower()

        if t in _SKIP_TERMS or not t:
            return None

        # Numeric range: "70-75", "100–110", "80–85" (dash or en-dash)
        range_match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)', t)
        if range_match:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            return ("range", (min(lo, hi), max(lo, hi)))

        # Plain number
        try:
            return ("exact", float(t))
        except ValueError:
            pass

        # Blanket term
        if t in _BLANKET_RANGES:
            return ("blanket", _BLANKET_RANGES[t])

        return None  # unrecognised — skip

    @staticmethod
    def _gaussian_reward(detected: float, target: float, sigma: float) -> float:
        return math.exp(-((detected - target) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def _trapezoid_reward(detected: float, lo: float, hi: float, ramp: float) -> float:
        if lo <= detected <= hi:
            return 1.0
        if lo - ramp <= detected < lo:
            return (detected - (lo - ramp)) / ramp
        if hi < detected <= hi + ramp:
            return ((hi + ramp) - detected) / ramp
        return 0.0

    def _best_octave(self, detected: float, score_fn) -> float:
        return max(score_fn(detected), score_fn(detected * 2), score_fn(detected / 2))

    def __compute_tempo_reward(self, detected_bpm: float, expected_tempo: str) -> Optional[float]:
        sigma = float(self.config.get("tempo_sigma", 6.0))
        ramp  = float(self.config.get("tempo_ramp_width", 15.0))

        parsed = self._parse_target_tempo(str(expected_tempo))
        if parsed is None:
            return None

        kind, value = parsed

        if kind == "exact":
            target = value
            return self._best_octave(detected_bpm, lambda d: self._gaussian_reward(d, target, sigma))

        if kind in ("range", "blanket"):
            lo, hi = value
            return self._best_octave(detected_bpm, lambda d: self._trapezoid_reward(d, lo, hi, ramp))
        
        
    def __krumhansl_profile(self, key: str, scale: str):
        import numpy as np
        MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        NOTE_TO_SEMITONE = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
            'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
        }
        profile = MAJOR if scale.lower() == 'major' else MINOR
        shift = NOTE_TO_SEMITONE.get(key, 0)
        return np.roll(profile, shift)


    def __compute_ks_key_reward(self, waveform, sample_rate, expected_key: str, expected_scale: str) -> float:
        import numpy as np
        import essentia.standard as es

        windowing = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        spectral_peaks = es.SpectralPeaks()
        hpcp_algo = es.HPCP()

        hpcp_frames = []
        for frame in es.FrameGenerator(self._to_essentia_array(waveform), frameSize=2048, hopSize=512):
            windowed = windowing(frame)
            spec = spectrum(windowed)
            freqs, mags = spectral_peaks(spec)
            hpcp_frames.append(hpcp_algo(freqs, mags))

        if not hpcp_frames:
            return 0.0

        mean_hpcp = np.mean(hpcp_frames, axis=0)
        target_profile = self.__krumhansl_profile(expected_key, expected_scale)

        corr = np.corrcoef(mean_hpcp, target_profile)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float((corr + 1) / 2)


    def __compute_key_reward(self, waveform_key, waveform_scale, waveform_strength, expected_key, expected_scale) -> float:
        import numpy as np

        COF = {'C':0,'G':1,'D':2,'A':3,'E':4,'B':5,
        'F#':6,'Gb':6,'C#':7,'Db':7,'G#':8,'Ab':8,
        'D#':9,'Eb':9,'A#':10,'Bb':10,'F':11}

        def key_distance(k1, s1, k2, s2):
            # Circular distance on circle of fifths
            d_tonic = min(abs(COF[k1] - COF[k2]), 12 - abs(COF[k1] - COF[k2]))
            # Mode penalty (relative major/minor are close, parallel are further)
            d_mode = 0 if s1 == s2 else 1
            return d_tonic + d_mode

        sigma = float(self.config.get("key_sigma", 2.0))

        def key_reward(pred, target, strength_pred, strength_target):
            d = key_distance(pred[0], pred[1], target[0], target[1])
            base = np.exp(-d**2 / (2 * sigma**2))   # Gaussian on distance
            return base * strength_pred * strength_target

        return key_reward((waveform_key, waveform_scale), (expected_key, expected_scale), waveform_strength, 1.0)
    
    
    def compute_rewards(self, waveform, sample_rate, expected_tempo, expected_key) -> Dict[str, Any]:
        pred_attributes = self.extract_attributes(waveform, sample_rate)
        tempo_reward = self.__compute_tempo_reward(pred_attributes["tempo"], expected_tempo)
        return {
            # None means the target had no tempo information — callers should skip this reward.
            "tempo": tempo_reward,
            "key": 0.5 * (self.__compute_key_reward(pred_attributes["key"]["key"], pred_attributes["key"]["scale"], pred_attributes["key"]["strength"], expected_key["key"], expected_key["scale"]) + self.__compute_ks_key_reward(waveform, sample_rate, expected_key["key"], expected_key["scale"])),
        }
        
