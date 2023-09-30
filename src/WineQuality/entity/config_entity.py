from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: Path
    target_column: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    x_train_data_path: Path
    y_train_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    x_test_data_path: Path
    y_test_data_path: Path
    model_path: Path
    metric_file_name: Path
    all_params: dict
    target_column: str
    mlflow_uri: str