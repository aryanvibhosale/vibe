from importlib import import_module

__all__ = [
    # configurator interface (reward_models.base)
    "RewardConfigurator",
    "build_reward_configurator",
    "available_reward_types",
    # reward_models.v2m_reward
    "JudgeRewardTensor",
    "JudgeModelInference",
    "JudgeRawScores",
    "JudgeScores",
    "aggregate_scores",
    "batch_zscore_to_r",
    # reward_models.hard_reward
    "HardVerifiableRewards",
    # per-family configurators
    "JudgeConfigurator",
    "CMIConfigurator",
    "HardVerifiableConfigurator",
]

_EXPORTS = {
    "RewardConfigurator": "reward_models.base",
    "build_reward_configurator": "reward_models.base",
    "available_reward_types": "reward_models.base",
    "JudgeConfigurator": "reward_models.v2m_reward.configurator",
    "CMIConfigurator": "reward_models.soft_reward.configurator",
    "HardVerifiableConfigurator": "reward_models.hard_reward.configurator",
    "JudgeRewardTensor": "reward_models.v2m_reward",
    "JudgeModelInference": "reward_models.v2m_reward",
    "JudgeRawScores": "reward_models.v2m_reward",
    "JudgeScores": "reward_models.v2m_reward",
    "aggregate_scores": "reward_models.v2m_reward",
    "batch_zscore_to_r": "reward_models.v2m_reward",
    "HardVerifiableRewards": "reward_models.hard_reward",
}


def __getattr__(name):                      # PEP 562
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
