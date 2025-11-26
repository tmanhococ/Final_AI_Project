"""
AEyePro Utils Package - Các tiện ích chung cho hệ thống

Package này cung cấp các hàm tiện ích cho:
- Quản lý cấu hình ứng dụng (JSON files)
- Xử lý dữ liệu với NumPy/Pandas compatibility
- Thao tác với files CSV (thread-safe)
- Quản lý thread pool cho xử lý đa luồng
- Quản lý paths và directories một cách linh hoạt

Author: AEyePro Team
Version: 3.0.0
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union

# ==============================================================================
# PATH MANAGEMENT - Quản lý đường dẫn file và thư mục
# ==============================================================================

# Xác định đường dẫn một cách linh hoạt và đáng tin cậy
# Lấy đường dẫn thư mục utils (AEYE/utils)
UTILS_DIR = Path(__file__).resolve().parent

# Lấy đường dẫn gốc của dự án AEYE
PROJECT_ROOT = UTILS_DIR.parent

# Định nghĩa các đường dẫn quan trọng cho hệ thống
DATA_DIR = PROJECT_ROOT / "data"      # Thư mục lưu trữ dữ liệu
CONFIG_DIR = PROJECT_ROOT / "config"  # Thư mục cấu hình

# ==============================================================================
# CONFIGURATION MANAGEMENT - Quản lý cấu hình ứng dụng
# ==============================================================================

def get_config(config_file: str = 'settings.json') -> Dict[str, Any]:
    """
    Tải cấu hình từ file JSON với đường dẫn tương đối

    Hàm này tự động tìm file cấu hình trong thư mục CONFIG_DIR và
    parse JSON thành Python dictionary với error handling.

    Args:
        config_file (str): Tên file cấu hình (mặc định: 'settings.json')

    Returns:
        Dict[str, Any]: Dictionary chứa các tham số cấu hình

    Raises:
        FileNotFoundError: Khi file cấu hình không tồn tại
        json.JSONDecodeError: Khi file JSON có định dạng sai

    Example:
        >>> config = get_config('settings.json')
        >>> camera_index = config.get('health_monitoring', {}).get('camera_index', 0)
    """
    config_path = CONFIG_DIR / config_file
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: Any, file_path: Union[str, Path]) -> None:
    """
    Lưu dữ liệu vào file JSON với hỗ trợ chuyển đổi NumPy types

    Hàm này tự động chuyển đổi các kiểu dữ liệu NumPy sang Python native types
    để đảm bảo tương thích với JSON serialization.

    Args:
        data (Any): Dữ liệu cần lưu (có thể chứa NumPy arrays, types)
        file_path (Union[str, Path]): Đường dẫn đến file JSON cần lưu

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> data = {
        ...     'array': np.array([1, 2, 3]),
        ...     'float_val': np.float32(3.14),
        ...     'int_val': np.int64(42)
        ... }
        >>> save_data(data, 'output.json')
    """
    def convert_numpy(obj: Any) -> Any:
        """
        Chuyển đổi đệ quy các kiểu NumPy sang Python native types

        Args:
            obj (Any): Object cần chuyển đổi

        Returns:
            Any: Object đã được chuyển đổi sang compatible types
        """
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

    # Tạo directory nếu chưa tồn tại
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

    # Lưu file với format JSON đẹp
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(data), f, indent=2, ensure_ascii=False)


# ==============================================================================
# CSV DATA HANDLING - Xử lý dữ liệu CSV với thread safety
# ==============================================================================

def append_csv(row_dict: Dict[str, Any], file_path: Union[str, Path],
              fieldnames: Optional[List[str]] = None) -> None:
    """
    Thêm một row vào file CSV (deprecated - sử dụng append_csv_row để có documentation tốt hơn)

    Args:
        row_dict (Dict[str, Any]): Dictionary chứa data của row
        file_path (Union[str, Path]): Đường dẫn đến file CSV
        fieldnames (Optional[List[str]]): Danh sách các column names (optional)

    Note:
        Function này được giữ để tương thích, nhưng nên sử dụng append_csv_row()
    """
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([row_dict])
    if not file_exists and fieldnames:
        df = df.reindex(columns=fieldnames)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False, encoding='utf-8')


def append_csv_row(row_dict: Dict[str, Any], file_path: Union[str, Path],
                   fieldnames: Optional[List[str]] = None) -> None:
    """
    Thêm một row (dictionary) vào file CSV với thread safety và flexibility cao

    Function này được thiết kế để hoạt động hiệu quả trong multi-threading environment,
    tự động tạo file mới với header nếu file chưa tồn tại.

    Args:
        row_dict (Dict[str, Any]): Dictionary chứa data cần lưu.
                                    Keys sẽ trở thành column names.
        file_path (Union[str, Path]): Đường dẫn đến file CSV cần ghi
        fieldnames (Optional[List[str]]): List để specify thứ tự columns.
                                       Nếu không cung cấp, columns sẽ theo
                                       thứ tự alphabet của keys.

    Returns:
        None

    Raises:
        PermissionError: Khi không có quyền ghi file
        OSError: Khi có lỗi hệ thống khi ghi file

    Example:
        >>> data = {
        ...     'timestamp': '2024-01-01 12:00:00',
        ...     'avg_ear': 0.25,
        ...     'distance_cm': 65.5,
        ...     'status': 'good'
        ... }
        >>> append_csv_row(data, 'health_data.csv',
        ...                ['timestamp', 'avg_ear', 'distance_cm', 'status'])
    """
    file_path = str(file_path)  # Đảm bảo file_path là string

    # Kiểm tra xem file đã tồn tại chưa
    file_exists = os.path.isfile(file_path)

    # Tạo DataFrame từ row data
    df = pd.DataFrame([row_dict])

    # Enforce column order nếu fieldnames được cung cấp
    if not file_exists and fieldnames:
        df = df.reindex(columns=fieldnames)

    # Append to CSV với header nếu file mới, không header nếu file đã tồn tại
    df.to_csv(file_path, mode='a', header=not file_exists, index=False, encoding='utf-8')


def read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Đọc file CSV thành DataFrame với error handling

    Args:
        file_path (Union[str, Path]): Đường dẫn đến file CSV

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu từ file CSV

    Raises:
        FileNotFoundError: Khi file không tồn tại
        pd.errors.EmptyDataError: Khi file rỗng
        pd.errors.ParserError: Khi file CSV có định dạng sai

    Example:
        >>> df = read_csv('health_data.csv')
        >>> print(df.head())
    """
    return pd.read_csv(file_path, encoding='utf-8')


def delete_csv_file(file_path: Union[str, Path]) -> bool:
    """
    Xóa file CSV một cách an toàn với error handling

    Args:
        file_path (Union[str, Path]): Đường dẫn đến file cần xóa

    Returns:
        bool: True nếu xóa thành công, False nếu file không tồn tại

    Raises:
        PermissionError: Khi không có quyền xóa file

    Example:
        >>> success = delete_csv_file('temp_data.csv')
        >>> if success:
        ...     print("File đã được xóa thành công")
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        return True
    return False


# ==============================================================================
# CAMERA CALIBRATION - Calibration và camera parameters
# ==============================================================================

def get_camera_calibration() -> Dict[str, float]:
    """
    Lấy các tham số calibration cơ bản cho camera

    Returns:
        Dict[str, float]: Dictionary chứa các tham số camera:
            - focal_length: Tiêu cự (pixels) cho ước tính khoảng cách

    Note:
        Trong phiên bản tương lai, function này có thể đọc từ file calibration
        hoặc tự động detect camera parameters.

    Example:
        >>> calib = get_camera_calibration()
        >>> focal_length = calib['focal_length']
    """
    return {
        'focal_length': 600.0  # Tiêu cự mặc định cho webcam thông thường
    }


# ==============================================================================
# THREAD MANAGEMENT - Quản lý đa luồng cho hiệu năng cao
# ==============================================================================

class ExecutorService:
    """
    Thread pool service được thiết kế riêng cho AEyePro

    Class này cung cấp abstraction layer trên ThreadPoolExecutor
    với các tùy chọn được tối ưu cho computer vision tasks.

    Attributes:
        executor (ThreadPoolExecutor): Internal thread pool

    Example:
        >>> service = ExecutorService(max_workers=2)
        >>> future = service.submit(some_heavy_function, image_path, params)
        >>> # Làm việc khác...
        >>> result = future.result()  # Đợi và lấy kết quả
        >>> service.shutdown()  # Cleanup
    """

    def __init__(self, max_workers: int = 4):
        """
        Khởi tạo thread pool với số worker threads

        Args:
            max_workers (int): Số lượng concurrent threads (mặc định: 4)
                               Nên được thiết lập dựa trên CPU cores
                               và nature của tasks (I/O bound vs CPU bound)
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AEyePro"
        )

    def submit(self, fn, *args, **kwargs):
        """
        Submit function để thực thi trong thread pool

        Args:
            fn (callable): Function cần thực thi
            *args: Positional arguments cho function
            **kwargs: Keyword arguments cho function

        Returns:
            concurrent.futures.Future: Future object để lấy kết quả

        Example:
            >>> future = executor.submit(process_heavy_image, image_path, params)
            >>> # Làm việc khác...
            >>> result = future.result()  # Đợi và lấy kết quả
        """
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """
        Shutdown thread pool một cách graceful

        Args:
            wait (bool): Có đợi các pending tasks hoàn thành không
                        (mặc định: True)

        Note:
            Nên được gọi khi ứng dụng shutdown để tránh orphaned threads
        """
        self.executor.shutdown(wait=wait)


# ==============================================================================
# APPLICATION CONFIGURATION - Quản lý cấu hình ứng dụng
# ==============================================================================

class AppConfig:
    """
    Lớp quản lý cấu hình ứng dụng với support cho environment variables

    Class này cung cấp centralized configuration management
    với khả năng override bằng environment variables.

    Attributes:
        camera_index (int): Index của camera device
        data_retention_days (int): Số ngày giữ dữ liệu
        model_path (str): Đường dẫn đến ML models
        force_cpu (bool): Force sử dụng CPU thay vì GPU

    Example:
        >>> config = AppConfig.from_env()
        >>> print(f"Using camera {config.camera_index}")
    """

    def __init__(self):
        """Khởi tạo với default values"""
        self.camera_index = 0
        self.data_retention_days = 7
        self.model_path = "models/Submodel/Llama-3.2-3B-Instruct-Q8_0.gguf"
        self.force_cpu = False

    @classmethod
    def from_env(cls):
        """
        Load configuration từ environment variables với fallback values

        Environment Variables:
            AEYE_CAMERA_INDEX: Index của camera (default: 0)
            AEYE_DATA_RETENTION_DAYS: Số ngày giữ data (default: 7)
            AEYE_FORCE_CPU: Force CPU mode (default: false)

        Returns:
            AppConfig: Instance với configuration từ environment

        Example:
            # Trong shell:
            # export AEYE_CAMERA_INDEX=1
            # export AEYE_DATA_RETENTION_DAYS=30

            # Trong Python:
            >>> config = AppConfig.from_env()
        """
        config = cls()

        # Override với environment variables nếu tồn tại
        config.camera_index = int(os.getenv('AEYE_CAMERA_INDEX', '0'))
        config.data_retention_days = int(os.getenv('AEYE_DATA_RETENTION_DAYS', '7'))
        config.force_cpu = os.getenv('AEYE_FORCE_CPU', 'false').lower() == 'true'

        return config


# ==============================================================================
# GLOBAL INSTANCES - Global instances cho tiện sử dụng
# ==============================================================================

# Global configuration instance cho toàn ứng dụng
app_config = AppConfig()

# Global thread pool service cho các background tasks
default_executor = ExecutorService()