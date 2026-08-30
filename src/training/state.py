from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingState:

    generator: object
    optimizer: object
    scheduler: object
    train_loader: object
    val_loader: object
    tracker: object
    batch_processor: object

