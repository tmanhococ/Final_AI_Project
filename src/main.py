"""
AEYE Vision Module - Hệ Thống Theo Dõi Sức Khỏe Mắt và Tư Thế (Vision Only)
Entry Point chỉ tập trung vào Computer Vision module

File này khởi động vision module với:
- Eye tracking với MediaPipe
- Posture analysis
- Blink detection
- Drowsiness detection
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from utils import get_config
    from vision import EyeTracker, PostureAnalyzer, BlinkDetector, HealthDataCollector
except ImportError as e:
    print(f"LOI import module: {e}")
    print("Kiem tra lai cau truc thu muc va dependencies!")
    sys.exit(1)


def setup_environment():
    """
    Thiết lập môi trường chạy ứng dụng
    """
    # Set environment variables
    os.environ["AEYE_VERSION"] = "2.0.0"
    os.environ["AEYE_MODE"] = "production"

    # Create data directory if not exists
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create config directory if not exists
    config_dir = current_dir / "config"
    config_dir.mkdir(exist_ok=True)


def check_dependencies():
    """
    Kiểm tra các dependencies cần thiết cho vision module
    """
    required_packages = [
        "cv2",           # OpenCV
        "mediapipe",     # MediaPipe
        "numpy",         # NumPy
        "pandas",        # Pandas
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("THIEU cac package sau:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nVui long chay: pip install -r requirements.txt")
        return False

    return True


def main():
    """
    Hàm chính của Vision Module
    """
    print("KHOI DONG AEYE Vision Module...")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Check dependencies
    if not check_dependencies():
        input("Nhan Enter de thoat...")
        return 1

    try:
        # Load configuration
        config = get_config()
        print("DA tai cau hình thanh cong")

        # Initialize vision modules
        print("KHOI TAO vision modules...")
        eye_tracker = EyeTracker()
        posture_analyzer = PostureAnalyzer()

        print("AEYE Vision Module da san sang!")
        print("Eye tracking và posture analysis ready")
        print("Nhan Ctrl+C de thoat")
        print("=" * 50)

        # Simple demonstration - run for 30 seconds
        import time
        start_time = time.time()

        try:
            while time.time() - start_time < 30:  # Run for 30 seconds
                # Simulate vision processing
                time.sleep(1)
                print(f"Vision processing... ({int(time.time() - start_time)}s)")
        except KeyboardInterrupt:
            print("\nTam biet AEYE Vision Module!")
            return 0

    except KeyboardInterrupt:
        print("\nTam biet AEYE Vision Module!")
        return 0
    except Exception as e:
        print(f"LOI khong mong muon: {e}")
        print("\nChi tiet loi:")
        import traceback
        traceback.print_exc()
        input("Nhan Enter de thoat...")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)