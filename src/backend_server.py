"""
AEyePro Backend Server - Flask API and WebSocket Bridge

This module provides the backend infrastructure to connect the Python-based
Computer Vision Health Monitoring System (vision_app.py) with the Web UI.

Architecture:
- Flask HTTP Server: Handles REST API endpoints for control and configuration
- Flask-SocketIO: Manages WebSocket connections for real-time data streaming
- VisionManager: Encapsulates vision processing logic and thread management
- Thread Safety: Uses locks to prevent race conditions between threads

Author: AEyePro Team
Version: 1.0.0
"""

import sys
import json
import time
import threading
import base64
import traceback
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import vision manager
from vision.vision_manager import VisionManager


# =============================================================================
# PART 1: BACKEND API SETUP
# Flask Server Configuration with SocketIO and CORS
# =============================================================================

# Initialize Flask application
app = Flask(__name__, static_folder='ui_module', static_url_path='')

# Configure Flask application settings
app.config['SECRET_KEY'] = 'aeyepro_secret_key_2025'
app.config['JSON_SORT_KEYS'] = False

# Enable Cross-Origin Resource Sharing (CORS) for all routes
# This allows the web UI to communicate with the backend from different origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Flask-SocketIO for WebSocket communication
# Using threading async_mode for compatibility with Flask development server
# In production, consider using eventlet or gevent for better performance
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,  # Tắt emit logging
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)


# =============================================================================
# VISION MANAGER INSTANCE
# =============================================================================

# Create a single global instance of VisionManager
# This instance will be shared across all Flask routes and SocketIO handlers
vision_manager = VisionManager()


# =============================================================================
# REAL-TIME STREAMING
# WebSocket Handlers for Frame and Metrics Broadcasting
# =============================================================================

def broadcast_frame_loop():
    """
    Continuously broadcast camera frames to all connected WebSocket clients
    
    This function runs in a background thread and periodically emits
    base64-encoded JPEG frames to all connected clients. The frames are
    encoded to reduce bandwidth and ensure compatibility with web browsers.
    
    Broadcasting Strategy:
    - Runs at ~15 FPS to reduce network load (half of processing rate)
    - Encodes frames as JPEG with 85% quality for size/quality balance
    - Uses socketio.emit() to broadcast to all connected clients
    - Handles cases where no frame is available gracefully
    
    Thread Safety:
        Accesses vision_manager.get_latest_frame() which is thread-safe
    """
    print("[Broadcast] Frame streaming thread started")
    
    while True:
        try:
            # Check if vision system is running
            status = vision_manager.get_status()
            if not status['is_running']:
                # Sleep longer when not running to reduce CPU usage
                time.sleep(0.5)
                continue
            
            # Get latest frame from vision manager (thread-safe)
            frame = vision_manager.get_latest_frame()
            
            if frame is not None:
                # Encode frame as JPEG for efficient transmission
                # Quality=85 provides good balance between size and visual quality
                encode_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if encode_success:
                    # Convert to base64 string for JSON transmission
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Broadcast to all connected clients via WebSocket
                    # namespace='/' is the default namespace
                    socketio.emit('camera_frame', {
                        'frame': frame_base64,
                        'timestamp': time.time()
                    }, namespace='/')
            
            # Maintain ~15 FPS broadcast rate (every 66ms)
            time.sleep(0.066)
            
        except Exception as e:
            print(f"[Broadcast] Error in frame loop: {e}")
            time.sleep(0.1)


def broadcast_metrics_loop():
    """
    Continuously broadcast health metrics to all connected WebSocket clients
    
    This function runs in a background thread and periodically emits
    health metrics (EAR, blink rate, posture, drowsiness) to all connected
    clients for real-time dashboard updates.
    
    Broadcasting Strategy:
    - Runs at ~2 Hz (every 0.5s) to provide smooth updates without overwhelming network
    - Sends JSON data directly (no encoding needed)
    - Includes all metrics from vision modules
    
    Thread Safety:
        Accesses vision_manager.get_latest_metrics() which is thread-safe
    """
    print("[Broadcast] Metrics streaming thread started")
    
    while True:
        try:
            # Check if vision system is running
            status = vision_manager.get_status()
            if not status['is_running']:
                # Sleep longer when not running
                time.sleep(1.0)
                continue
            
            # Get latest health metrics from vision manager (thread-safe)
            metrics = vision_manager.get_latest_metrics()
            
            if metrics:
                # Broadcast metrics to all connected clients
                socketio.emit('health_metrics', metrics, namespace='/')
            
            # Update every 0.5 seconds (2 Hz)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"[Broadcast] Error in metrics loop: {e}")
            time.sleep(0.5)


@socketio.on('connect')
def handle_connect():
    """
    Handle new WebSocket client connection
    
    This event fires when a client successfully establishes a WebSocket
    connection to the server. We can use this to send initial state
    or perform connection logging.
    """
    print(f"[WebSocket] Client connected: {request.sid}")
    
    # Send current system status to newly connected client
    status = vision_manager.get_status()
    emit('system_status', status)


@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle WebSocket client disconnection
    
    This event fires when a client disconnects from the WebSocket.
    Useful for cleanup and logging.
    """
    print(f"[WebSocket] Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    """
    Handle client request for current system status
    
    Clients can emit 'request_status' to get the current state
    of the vision system on demand.
    """
    status = vision_manager.get_status()
    emit('system_status', status)


# =============================================================================
# REST API ROUTES
# REST API Routes for Camera Control and Settings Management
# =============================================================================

@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    """
    Start the camera and vision processing system
    
    POST /api/camera/start
    
    Request Body: None (optional JSON with config overrides)
    
    Response:
        {
            "success": true/false,
            "message": "Status message",
            "session_id": "session_identifier",
            "error": "error_code" (if success is false)
        }
    
    Error Codes:
        - ALREADY_RUNNING: Vision system is already active
        - INITIALIZATION_FAILED: Failed to initialize vision modules
        - CAMERA_UNAVAILABLE: Camera device could not be opened
        - EXCEPTION: Unexpected error occurred
    
    Example:
        POST /api/camera/start
        Response: {"success": true, "message": "Vision system started", "session_id": "abc123"}
    """
    try:
        print("[API] POST /api/camera/start - Starting camera")
        
        # Start the vision manager
        result = vision_manager.start()
        
        # Return appropriate HTTP status code based on result
        status_code = 200 if result['success'] else 400
        
        return jsonify(result), status_code
        
    except Exception as e:
        print(f"[API] Error in camera start: {e}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error": "SERVER_ERROR",
            "details": traceback.format_exc()
        }), 500


@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    """
    Stop the camera and vision processing system
    
    POST /api/camera/stop
    
    Request Body: None
    
    Response:
        {
            "success": true/false,
            "message": "Status message",
            "error": "error_code" (if success is false)
        }
    
    Error Codes:
        - NOT_RUNNING: Vision system is not currently active
        - EXCEPTION: Error occurred during shutdown
    
    Example:
        POST /api/camera/stop
        Response: {"success": true, "message": "Vision system stopped"}
    """
    try:
        print("[API] POST /api/camera/stop - Stopping camera")
        
        # Stop the vision manager
        result = vision_manager.stop()
        
        # Return appropriate HTTP status code based on result
        status_code = 200 if result['success'] else 400
        
        return jsonify(result), status_code
        
    except Exception as e:
        print(f"[API] Error in camera stop: {e}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error": "SERVER_ERROR",
            "details": traceback.format_exc()
        }), 500


@app.route('/api/camera/status', methods=['GET'])
def api_camera_status():
    """
    Get current status of the vision system
    
    GET /api/camera/status
    
    Response:
        {
            "is_running": true/false,
            "session_id": "session_identifier" or null,
            "fps": 30.5,
            "last_error": "error_message" or null
        }
    
    Example:
        GET /api/camera/status
        Response: {"is_running": true, "session_id": "abc123", "fps": 30.2, "last_error": null}
    """
    try:
        status = vision_manager.get_status()
        return jsonify(status), 200
        
    except Exception as e:
        print(f"[API] Error getting camera status: {e}")
        return jsonify({
            "error": "SERVER_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """
    Retrieve current settings from config/settings.json
    
    GET /api/settings
    
    Response:
        {
            "success": true,
            "settings": {
                "health_monitoring": {...},
                "ui_settings": {...}
            }
        }
    
    Error Codes:
        - FILE_NOT_FOUND: settings.json does not exist
        - PARSE_ERROR: JSON parsing failed
        - SERVER_ERROR: Unexpected error
    
    Example:
        GET /api/settings
        Response: {"success": true, "settings": {...}}
    """
    try:
        print("[API] GET /api/settings - Reading configuration")
        
        # Construct path to settings.json
        config_path = Path(current_dir) / "config" / "settings.json"
        
        # Check if file exists
        if not config_path.exists():
            return jsonify({
                "success": False,
                "message": "Settings file not found",
                "error": "FILE_NOT_FOUND"
            }), 404
        
        # Read and parse JSON file
        with open(config_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        return jsonify({
            "success": True,
            "settings": settings
        }), 200
        
    except json.JSONDecodeError as e:
        print(f"[API] JSON parsing error: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to parse settings file",
            "error": "PARSE_ERROR",
            "details": str(e)
        }), 400
        
    except Exception as e:
        print(f"[API] Error reading settings: {e}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error": "SERVER_ERROR"
        }), 500


@app.route('/api/settings/face-mesh', methods=['POST'])
def api_toggle_face_mesh():
    """
    Toggle face mesh landmarks visibility
    
    POST /api/settings/face-mesh
    Body: {"enabled": true/false}
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        # Update vision app's face mesh setting
        if vision_manager.vision_app:
            vision_manager.vision_app.show_face_mesh = enabled
            
        return jsonify({
            "success": True,
            "face_mesh_enabled": enabled
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/settings', methods=['POST'])
def api_update_settings():
    """
    Update settings in config/settings.json and reload vision system
    
    POST /api/settings
    
    Request Body:
        {
            "settings": {
                "health_monitoring": {...},
                "ui_settings": {...}
            },
            "reload_vision": true/false (optional, default: true)
        }
    
    Response:
        {
            "success": true/false,
            "message": "Status message",
            "reloaded": true/false,
            "error": "error_code" (if success is false)
        }
    
    Behavior:
        1. Validates JSON structure
        2. Backs up current settings
        3. Writes new settings to file
        4. If vision system is running and reload_vision=true:
           - Stops current vision system
           - Starts new vision system with updated settings
    
    Error Codes:
        - INVALID_JSON: Request body is not valid JSON
        - MISSING_SETTINGS: Required 'settings' field missing
        - WRITE_ERROR: Failed to write to settings file
        - RELOAD_ERROR: Settings saved but reload failed
        - SERVER_ERROR: Unexpected error
    
    Example:
        POST /api/settings
        Body: {"settings": {...}, "reload_vision": true}
        Response: {"success": true, "message": "Settings updated", "reloaded": true}
    """
    try:
        print("[API] POST /api/settings - Updating configuration")
        
        # Parse request JSON
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                "success": False,
                "message": "Request body must be valid JSON",
                "error": "INVALID_JSON"
            }), 400
        
        # Extract settings from request
        new_settings = request_data.get('settings')
        reload_vision = request_data.get('reload_vision', True)
        
        if not new_settings:
            return jsonify({
                "success": False,
                "message": "Missing 'settings' field in request body",
                "error": "MISSING_SETTINGS"
            }), 400
        
        # Construct path to settings.json
        config_path = Path(current_dir) / "config" / "settings.json"
        
        # Create backup of current settings before overwriting
        backup_path = config_path.with_suffix('.json.backup')
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"[API] Created settings backup: {backup_path}")
        
        # Write new settings to file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_settings, f, indent=2, ensure_ascii=False)
        
        print("[API] Settings file updated successfully")
        
        # Hot-reload vision system if requested and currently running
        vision_reloaded = False
        reload_error = None
        
        if reload_vision:
            status = vision_manager.get_status()
            
            if status['is_running']:
                print("[API] Reloading vision system with new settings...")
                
                # Stop current vision system
                stop_result = vision_manager.stop()
                
                if stop_result['success']:
                    # Brief pause to ensure clean shutdown
                    time.sleep(0.2)
                    
                    # Start vision system with new settings
                    start_result = vision_manager.start()
                    
                    if start_result['success']:
                        vision_reloaded = True
                        print("[API] Vision system reloaded successfully")
                    else:
                        reload_error = start_result.get('message', 'Failed to restart vision system')
                        print(f"[API] Failed to restart vision system: {reload_error}")
                else:
                    reload_error = stop_result.get('message', 'Failed to stop vision system')
                    print(f"[API] Failed to stop vision system: {reload_error}")
        
        # Prepare response
        response = {
            "success": True,
            "message": "Settings updated successfully",
            "reloaded": vision_reloaded
        }
        
        if reload_error:
            response["reload_warning"] = reload_error
        
        return jsonify(response), 200
        
    except json.JSONDecodeError as e:
        print(f"[API] JSON parsing error: {e}")
        return jsonify({
            "success": False,
            "message": "Invalid JSON in request body",
            "error": "INVALID_JSON",
            "details": str(e)
        }), 400
        
    except IOError as e:
        print(f"[API] File write error: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to write settings file",
            "error": "WRITE_ERROR",
            "details": str(e)
        }), 500
        
    except Exception as e:
        print(f"[API] Error updating settings: {e}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}",
            "error": "SERVER_ERROR",
            "details": traceback.format_exc()
        }), 500


# =============================================================================
# FLASK ROUTES - Static File Serving
# =============================================================================

@app.route('/')
def serve_index():
    """
    Serve the main HTML page
    
    Returns:
        HTML content of index.html from ui_module directory
    """
    return send_from_directory('ui_module', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """
    Serve static files (CSS, JS, images, etc.)
    
    Args:
        path: Relative path to the static file
        
    Returns:
        Requested static file from ui_module directory
    """
    return send_from_directory('ui_module', path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    """
    Main entry point for the backend server
    
    This starts the Flask-SocketIO server in development mode and launches
    background threads for real-time data streaming.
    
    For production deployment, use a production WSGI server like Gunicorn
    with eventlet or gevent workers.
    """
    print("=" * 70)
    print("AEyePro Backend Server - Starting")
    print("=" * 70)
    print("Server URL: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    # Start background threads for real-time streaming
    # These threads will continuously broadcast frames and metrics to connected clients
    frame_thread = threading.Thread(
        target=broadcast_frame_loop,
        name="FrameBroadcastThread",
        daemon=True
    )
    frame_thread.start()
    
    metrics_thread = threading.Thread(
        target=broadcast_metrics_loop,
        name="MetricsBroadcastThread",
        daemon=True
    )
    metrics_thread.start()
    
    print("[Server] Background streaming threads started")
    print("  - Frame broadcast: ~15 FPS")
    print("  - Metrics broadcast: ~2 Hz")
    print("=" * 70)
    
    # Start Flask-SocketIO server
    # debug=True enables auto-reload on code changes (development only)
    # use_reloader=False prevents double initialization in debug mode
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
