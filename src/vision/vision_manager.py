"""
Vision Manager Module - Manages Vision Processing in Separate Thread

This module encapsulates the VisionManager class which manages the lifecycle
of the AEyeProVisionApp and ensures vision processing runs in a separate thread.

Author: AEyePro Team
Version: 1.0.0
"""

import time
import threading
import traceback
from typing import Dict, Any, Optional

import cv2
import numpy as np

from vision.vision_app import AEyeProVisionApp


class VisionManager:
    """
    Vision Manager - Encapsulates vision processing logic and thread management
    
    This class manages the lifecycle of the AEyeProVisionApp and ensures that
    vision processing runs in a separate thread without blocking the Flask server.
    
    Key Responsibilities:
    - Initialize and manage AEyeProVisionApp instance
    - Run vision processing loop in a background thread
    - Provide thread-safe access to vision data and camera frames
    - Handle graceful shutdown and resource cleanup
    
    Thread Safety:
    - Uses threading.Lock() to protect shared state variables
    - Ensures atomic operations when starting/stopping the vision loop
    - Prevents race conditions between Flask routes and vision thread
    """
    
    def __init__(self, config_path: str = "settings.json"):
        """
        Initialize Vision Manager with configuration
        
        Args:
            config_path: Path to settings.json configuration file (relative to config directory)
        """
        # Configuration file path
        self.config_path = config_path
        
        # Vision application instance (initialized on start)
        self.vision_app: Optional[AEyeProVisionApp] = None
        
        # Thread management
        self.vision_thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        
        # Thread safety lock to prevent race conditions
        # This lock protects: is_running, vision_app, vision_thread
        self.lock = threading.Lock()
        
        # Latest processed data from vision modules
        # Access to these should be protected by the lock
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_health_metrics: Dict[str, Any] = {}
        
        # Error tracking
        self.last_error: Optional[str] = None
        
        # Frame processing statistics
        self.frame_count: int = 0
        self.frames_per_second: float = 0.0
        self.last_fps_update_time: float = time.time()
        
    def start(self) -> Dict[str, Any]:
        """
        Start the vision processing system
        
        This method initializes the AEyeProVisionApp and starts the vision
        processing loop in a separate background thread. The thread runs
        continuously until stop() is called.
        
        Returns:
            Dict containing status information:
                - success: Boolean indicating if start was successful
                - message: Human-readable status message
                - error: Error details if success is False
                
        Thread Safety:
            Uses self.lock to ensure atomic state transitions
        """
        with self.lock:
            # Check if already running
            if self.is_running:
                return {
                    "success": False,
                    "message": "Vision system is already running",
                    "error": "ALREADY_RUNNING"
                }
            
            try:
                # Initialize AEyeProVisionApp with camera display disabled
                # The camera feed will be streamed via WebSocket instead
                self.vision_app = AEyeProVisionApp(
                    config_file=self.config_path,
                    show_camera=False  # Disable OpenCV window, use WebSocket
                )
                
                # Initialize all vision modules (eye tracker, posture analyzer, etc.)
                initialization_success = self.vision_app.initialize_modules()
                
                if not initialization_success:
                    self.last_error = "Failed to initialize vision modules"
                    return {
                        "success": False,
                        "message": "Failed to initialize vision modules",
                        "error": "INITIALIZATION_FAILED"
                    }
                
                # Setup session logging for data collection
                session_id = self.vision_app.setup_session_logging()
                
                # Start health data collector
                if self.vision_app.health_collector:
                    self.vision_app.health_collector.start_collection()
                    print(f"[OK] Health Data Collector started - Session: {session_id}")
                
                # Set start_time for session tracking
                self.vision_app.start_time = time.time()
                self.vision_app.running = True
                
                # Mark as running before starting thread
                self.is_running = True
                
                # Create and start the vision processing thread
                # daemon=True ensures thread terminates when main program exits
                self.vision_thread = threading.Thread(
                    target=self._vision_loop,
                    name="VisionProcessingThread",
                    daemon=True
                )
                self.vision_thread.start()
                
                # Reset error state
                self.last_error = None
                
                return {
                    "success": True,
                    "message": "Vision system started successfully",
                    "session_id": session_id
                }
                
            except Exception as e:
                # Cleanup on error
                self.is_running = False
                self.vision_app = None
                self.last_error = str(e)
                
                return {
                    "success": False,
                    "message": f"Failed to start vision system: {str(e)}",
                    "error": "EXCEPTION",
                    "details": traceback.format_exc()
                }
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop the vision processing system gracefully
        
        This method signals the vision thread to stop, waits for it to finish,
        and releases all resources including the camera.
        
        Returns:
            Dict containing status information:
                - success: Boolean indicating if stop was successful
                - message: Human-readable status message
                
        Thread Safety:
            Uses self.lock to ensure atomic state transitions
        """
        with self.lock:
            # Check if not running
            if not self.is_running:
                return {
                    "success": False,
                    "message": "Vision system is not running",
                    "error": "NOT_RUNNING"
                }
            
            try:
                # Signal the vision loop to stop
                self.is_running = False
                
                # Wait for the vision thread to finish (with timeout)
                if self.vision_thread and self.vision_thread.is_alive():
                    # Release the lock temporarily while waiting for thread
                    # to prevent deadlock if thread is trying to acquire lock
                    threading.Thread(
                        target=self._wait_for_thread_shutdown,
                        daemon=True
                    ).start()
                
                # Shutdown vision application and release resources
                if self.vision_app:
                    # Stop health data collector
                    if hasattr(self.vision_app, 'health_collector') and self.vision_app.health_collector:
                        self.vision_app.health_collector.stop_collection()
                        print("[OK] Health Data Collector stopped")
                    
                    # Call shutdown which will save summary and cleanup
                    if hasattr(self.vision_app, 'shutdown'):
                        self.vision_app.shutdown()
                    
                    # Stop eye tracker to release camera
                    if hasattr(self.vision_app, 'eye_tracker') and self.vision_app.eye_tracker:
                        self.vision_app.eye_tracker.stop()
                    
                    self.vision_app = None
                
                # Clear cached data
                self.latest_frame = None
                self.latest_health_metrics = {}
                
                return {
                    "success": True,
                    "message": "Vision system stopped successfully"
                }
                
            except Exception as e:
                self.last_error = str(e)
                return {
                    "success": False,
                    "message": f"Error stopping vision system: {str(e)}",
                    "error": "EXCEPTION",
                    "details": traceback.format_exc()
                }
    
    def _wait_for_thread_shutdown(self):
        """
        Helper method to wait for vision thread shutdown with timeout
        
        This runs in a separate thread to avoid blocking the stop() method
        """
        if self.vision_thread:
            self.vision_thread.join(timeout=5.0)
    
    def _vision_loop(self):
        """
        Main vision processing loop - runs in background thread
        
        This method continuously processes camera frames, extracts health metrics,
        and makes the data available for WebSocket streaming. It runs until
        is_running is set to False.
        
        Processing Pipeline:
        1. Process frame through all vision modules
        2. Extract camera frame for streaming
        3. Update statistics
        4. Save frame data
        5. Emit data via WebSocket (if connected)
        
        Thread Safety:
            Acquires self.lock only for brief updates to shared state
        """
        print("[VisionManager] Vision processing loop started")
        
        frame_id = 0
        
        while True:
            # Check if we should continue running (thread-safe check)
            with self.lock:
                if not self.is_running:
                    break
            
            try:
                # Process one frame through all vision modules
                # This calls eye_tracker, posture_analyzer, blink_detector, etc.
                frame_result = self.vision_app.process_frame()
                
                # Extract camera frame first
                raw_frame = self.vision_app.eye_tracker.get_frame()
                
                # Create annotated frame with landmarks if enabled
                if raw_frame is not None:
                    annotated_frame = raw_frame.copy()
                    
                    # Add error message if no face detected
                    if 'error' in frame_result:
                        h, w = annotated_frame.shape[:2]
                        cv2.putText(annotated_frame, "No Face Detected - Please position your face in view", 
                                  (int(w*0.1), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 0, 255), 2)
                    else:
                        # Draw face landmarks if enabled
                        if hasattr(self.vision_app, '_draw_face_landmarks'):
                            self.vision_app._draw_face_landmarks(annotated_frame, frame_result)
                else:
                    annotated_frame = None
                    
                # Skip statistics update if there was an error
                if 'error' in frame_result:
                    # Still update the frame for display, but skip data processing
                    with self.lock:
                        self.latest_frame = annotated_frame
                    time.sleep(0.033)  # ~30 FPS
                    continue
                
                # Update statistics from frame result
                self.vision_app.update_statistics(frame_result)
                
                # Save frame data for logging
                self.vision_app.save_frame_data(frame_result, frame_id)
                
                # Update shared state with latest data (thread-safe)
                with self.lock:
                    self.latest_frame = annotated_frame  # Use annotated frame instead of raw frame
                    self.latest_health_metrics = self._extract_health_metrics(frame_result)
                    self.frame_count += 1
                    
                    # Update FPS calculation every second
                    current_time = time.time()
                    if current_time - self.last_fps_update_time >= 1.0:
                        time_elapsed = current_time - self.last_fps_update_time
                        self.frames_per_second = self.frame_count / time_elapsed
                        self.frame_count = 0
                        self.last_fps_update_time = current_time
                
                frame_id += 1
                
                # Maintain processing rate at ~30 FPS
                time.sleep(0.033)
                
            except Exception as e:
                print(f"[VisionManager] Error in vision loop: {e}")
                with self.lock:
                    self.last_error = str(e)
                time.sleep(0.1)  # Brief pause before retry
        
        print("[VisionManager] Vision processing loop ended")
    
    def _extract_health_metrics(self, frame_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract health metrics from frame result for WebSocket transmission
        
        This method consolidates data from all vision modules into a single
        JSON-serializable dictionary for transmission to the web client.
        
        Args:
            frame_result: Processing result from vision_app.process_frame()
            
        Returns:
            Dict containing health metrics ready for JSON serialization
        """
        eye_data = frame_result.get('eye_data', {})
        blink_data = frame_result.get('blink_data', {})
        posture_data = frame_result.get('posture_data', {})
        drowsy_data = frame_result.get('drowsy_data', {})
        
        # Consolidate metrics into a structured format
        metrics = {
            # Timestamp
            'timestamp': time.time(),
            
            # Eye tracking metrics
            'eye': {
                'avg_ear': eye_data.get('avg_ear'),
                'left_ear': eye_data.get('left_ear'),
                'right_ear': eye_data.get('right_ear'),
                'distance_cm': eye_data.get('distance_cm'),
            },
            
            # Blink detection metrics
            'blink': {
                'total_blinks': blink_data.get('total_blinks', 0),
                'blink_detected': blink_data.get('blink_detected', False),
                'blink_rate': blink_data.get('blink_rate_per_minute', 0),
            },
            
            # Posture analysis metrics
            'posture': {
                'head_side_angle': posture_data.get('head_side_angle'),
                'head_updown_angle': posture_data.get('head_updown_angle'),
                'shoulder_tilt': posture_data.get('shoulder_tilt'),
                'eye_distance_cm': posture_data.get('eye_distance_cm'),
                'status': posture_data.get('status', 'unknown'),
            },
            
            # Drowsiness detection metrics
            'drowsiness': {
                'detected': drowsy_data.get('drowsiness_detected', False),
                'reason': drowsy_data.get('reason'),
                'ear_duration': drowsy_data.get('ear_duration', 0),
            },
            
            # System statistics
            'system': {
                'fps': self.frames_per_second,
                'session_id': self.vision_app.session_id if self.vision_app else None,
            }
        }
        
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the vision system
        
        Returns:
            Dict containing status information:
                - is_running: Boolean indicating if vision system is active
                - session_id: Current session ID (if running)
                - fps: Current frames per second
                - last_error: Last error message (if any)
        """
        with self.lock:
            return {
                'is_running': self.is_running,
                'session_id': self.vision_app.session_id if self.vision_app else None,
                'fps': self.frames_per_second,
                'last_error': self.last_error,
            }
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest processed camera frame (thread-safe)
        
        Returns:
            numpy array containing the latest frame, or None if not available
        """
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest health metrics (thread-safe)
        
        Returns:
            Dict containing the latest health metrics
        """
        with self.lock:
            return self.latest_health_metrics.copy()
