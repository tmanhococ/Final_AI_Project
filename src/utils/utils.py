import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Calculate paths more robustly
# Get the utils directory (AEYE/utils)
UTILS_DIR = Path(__file__).resolve().parent
# Get the AEYE project root
PROJECT_ROOT = UTILS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

def get_config(config_file='settings.json'):
    """Load configuration from JSON file."""
    with open(CONFIG_DIR / config_file, 'r') as f:
        return json.load(f)

def save_data(data, file_path):
    """Save data to JSON file with NumPy conversion."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj

    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(convert_numpy(data), f, indent=2)

def append_csv(row_dict, file_path, fieldnames=None):
    """Append row to CSV file."""
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([row_dict])
    if not file_exists and fieldnames:
        df = df.reindex(columns=fieldnames)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def append_csv_row(row_dict, file_path, fieldnames=None):
    """
    Append a row (dict) to a CSV file. Nếu file chưa tồn tại sẽ tạo mới với header.

    Args:
        row_dict: Dictionary chứa data cần lưu
        file_path: Đường dẫn đến file CSV
        fieldnames: List fieldnames để enforce column order (optional)
    """
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([row_dict])
    if not file_exists and fieldnames:
        df = df.reindex(columns=fieldnames)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)

def read_csv(file_path):
    """Read CSV file as DataFrame."""
    return pd.read_csv(file_path)

def delete_csv_file(file_path):
    """Xóa file CSV nếu tồn tại."""
    if os.path.isfile(file_path):
        os.remove(file_path)

def get_camera_calibration():
    """Get basic camera calibration."""
    return {'focal_length': 600}


class ExecutorService:
    """Simple thread pool executor for AEYE."""

    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AEye")

    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)


class AppConfig:
    """Application configuration management."""

    def __init__(self):
        self.camera_index = 0
        self.data_retention_days = 7
        self.model_path = "models/Submodel/Llama-3.2-3B-Instruct-Q8_0.gguf"
        self.force_cpu = False

    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        config = cls()
        config.camera_index = int(os.getenv('CAMERA_INDEX', '0'))
        config.data_retention_days = int(os.getenv('DATA_RETENTION_DAYS', '7'))
        config.force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        return config


app_config = AppConfig()