"""
Package UI - Giao diện người dùng cho AEYE

Package này chứa các module giao diện:
- Main window: Cửa sổ chính ứng dụng
- Health monitoring panel: Panel theo dõi sức khỏe
- AI assistant panel: Panel trợ lý AI
- Settings panel: Panel cài đặt
- Alert dialog: Dialog cảnh báo
"""

from .main_window import MainWindow
from .health_panel import HealthPanel
from .ai_assistant_panel import AIAssistantPanel
from .settings_panel import SettingsPanel
from .alert_dialog import AlertDialog

__all__ = [
    "MainWindow",
    "HealthPanel",
    "AIAssistantPanel",
    "SettingsPanel",
    "AlertDialog"
]