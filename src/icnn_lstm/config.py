from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # Paper-consistent topology: 3 sequential 1D CNN layers, then 3 parallel LSTM+attention blocks.
    input_features: int = 36
    cnn_filters: tuple[int, int, int] = (32, 32, 32)
    cnn_kernel_size: int = 9
    cnn_activation: str = "tanh"
    pool_size: int = 2

    lstm_units_per_block: tuple[int, int, int] = (384, 384, 384)
    num_parallel_lstm_blocks: int = 3
    attention_units: int = 64

    dense_units: int = 80
    second_dense_units: int = 2
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"
    dropout_rate: float = 0.10326648213511579
    learning_rate: float = 1e-3


@dataclass
class TrainingConfig:
    seed: int = 42
    initial_events: int = 40_000
    update_batch_events: int = 10_000
    update_train_ratio: float = 0.8
    update_epochs: int = 100
    initial_epochs: int = 100
    fit_batch_size: int = 1024
    early_stopping_patience: int = 3

    use_smote: bool = True
    threshold: float = 0.5

    # Mimics Algorithm 1 intent: freeze older layers and update outer layers on batch updates.
    freeze_base_on_update: bool = True


@dataclass
class DataConfig:
    train_csv: Path = Path("SILRAD-dataset/fasttext-trainmodel.csv")
    stream_csv: Path = Path("SILRAD-dataset/fasttext-testmodel.csv")
    label_col: str = "class"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: Path = Path("outputs")
    model_name: str = "icnn_lstm_plus.keras"
