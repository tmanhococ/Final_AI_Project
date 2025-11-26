"""
AEyePro Utils Package - Package initialization

Package này cung cấp các tiện ích chung cho toàn bộ hệ thống AEyePro,
bao gồm quản lý cấu hình, xử lý dữ liệu, và quản lý thread pool.

Main Components:
- Configuration Management: get_config(), AppConfig
- Data Storage: save_data(), append_csv_row(), read_csv()
- Path Management: DATA_DIR, CONFIG_DIR
- Thread Management: ExecutorService
- Camera Calibration: get_camera_calibration()

Usage:
    >>> from utils import get_config, app_config
    >>> config = get_config('settings.json')
    >>> camera_idx = app_config.camera_index

Author: AEyePro Team
Version: 3.0.0
"""

# Import tất cả các hàm và classes quan trọng từ utils.py
from .utils import (
    get_config,           # Load JSON configuration files
    save_data,           # Save data to JSON with NumPy conversion
    append_csv,          # Append to CSV (deprecated, use append_csv_row)
    append_csv_row,      # Append dictionary row to CSV with thread safety
    read_csv,            # Read CSV file as pandas DataFrame
    get_camera_calibration, # Get camera calibration parameters
    ExecutorService,      # Thread pool for background processing
    AppConfig,            # Application configuration class
    app_config,           # Global application configuration instance
    DATA_DIR,            # Path to data directory
    CONFIG_DIR            # Path to configuration directory
)

# Danh sách các symbols được export khi import *
__all__ = [
    # Configuration Functions
    'get_config',

    # Data Storage Functions
    'save_data',
    'append_csv',
    'append_csv_row',
    'read_csv',

    # Hardware Functions
    'get_camera_calibration',

    # Classes
    'ExecutorService',
    'AppConfig',

    # Global Instances
    'app_config',

    # Constants
    'DATA_DIR',
    'CONFIG_DIR'
]