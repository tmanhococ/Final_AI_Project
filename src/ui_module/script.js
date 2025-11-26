// AeyePro - Vision Module JavaScript

// ============================================================================
// CONFIGURATION
// ============================================================================

// Health metrics configuration
const HEALTH_METRICS_CONFIG = [
    { id: 'blinkRate', icon: '👁️', label: 'Blink Rate', defaultValue: '-- /min' },
    { id: 'avgEar', icon: '📊', label: 'Avg EAR', defaultValue: '--' },
    { id: 'avgDistance', icon: '📏', label: 'Distance', defaultValue: '-- cm' },
    { id: 'shoulderTilt', icon: '🤸', label: 'Shoulder', defaultValue: '--°' },
    { id: 'headPitch', icon: '🔽', label: 'Head Pitch', defaultValue: '--°' },
    { id: 'headYaw', icon: '↔️', label: 'Head Yaw', defaultValue: '--°' },
    { id: 'drowsyEvents', icon: '😴', label: 'Drowsy', defaultValue: '--' }
];

// Settings configuration - Simplified to 6 essential controls
const SETTINGS_CONFIG = [
    { id: 'frameRate', label: 'Frame Rate (FPS)', min: 10, max: 60, step: 1, default: 30, 
      description: 'Processing speed (lower = less CPU usage)' },
    { id: 'cameraIndex', label: 'Camera Index', min: 0, max: 3, step: 1, default: 0,
      description: '0=default, 1=external camera' },
    { id: 'drowsyThreshold', label: 'Drowsy EAR Threshold', min: 0.20, max: 0.40, step: 0.01, default: 0.30,
      description: 'Lower = more sensitive drowsiness detection' },
    { id: 'blinkThreshold', label: 'Blink EAR Threshold', min: 0.20, max: 0.35, step: 0.01, default: 0.27,
      description: 'Lower = more sensitive blink detection' },
    { id: 'minReasonableDistance', label: 'Min Distance (cm)', min: 15, max: 50, step: 5, default: 20,
      description: 'Minimum safe distance from screen' },
    { id: 'maxReasonableDistance', label: 'Max Distance (cm)', min: 80, max: 200, step: 10, default: 150,
      description: 'Maximum effective detection range' }
];

// Default settings values (generated from config)
const DEFAULT_SETTINGS = SETTINGS_CONFIG.reduce((acc, setting) => {
    acc[setting.id] = setting.default;
    return acc;
}, {});

// ============================================================================
// TEMPLATE RENDERING
// ============================================================================

/**
 * Creates a metric card element from template
 * @param {Object} config - Metric configuration
 * @returns {HTMLElement} Cloned and populated metric card
 */
function createMetricCard(config) {
    const template = document.getElementById('metricCardTemplate');
    const clone = template.content.cloneNode(true);

    clone.querySelector('[data-icon]').textContent = config.icon;
    clone.querySelector('[data-label]').textContent = config.label;
    clone.querySelector('[data-value]').textContent = config.defaultValue;
    clone.querySelector('[data-value]').id = config.id;

    return clone;
}

/**
 * Creates a setting input row from template
 * @param {Object} config - Setting configuration
 * @returns {HTMLElement} Cloned and populated setting input
 */
function createSettingInput(config) {
    const template = document.getElementById('settingInputTemplate');
    const clone = template.content.cloneNode(true);

    const label = clone.querySelector('[data-label]');
    const description = clone.querySelector('[data-description]');
    const input = clone.querySelector('[data-input]');
    const decrementBtn = clone.querySelector('[data-action="decrement"]');
    const incrementBtn = clone.querySelector('[data-action="increment"]');

    label.textContent = config.label;
    label.setAttribute('for', config.id);
    
    // Set description if available
    if (config.description && description) {
        description.textContent = config.description;
    }

    input.id = config.id;
    input.min = config.min;
    input.max = config.max;
    input.step = config.step;
    input.value = config.default;

    decrementBtn.dataset.target = config.id;
    decrementBtn.dataset.step = config.step;
    incrementBtn.dataset.target = config.id;
    incrementBtn.dataset.step = config.step;

    return clone;
}

/**
 * Initializes all UI components from configuration
 */
function initializeUI() {
    // Render health metrics
    const metricsContainer = document.getElementById('healthMetrics');
    HEALTH_METRICS_CONFIG.forEach(config => {
        metricsContainer.appendChild(createMetricCard(config));
    });

    // Render settings inputs
    const settingsGrid = document.getElementById('settingsGrid');
    SETTINGS_CONFIG.forEach(config => {
        settingsGrid.appendChild(createSettingInput(config));
    });

    // Add save button
    const saveBtn = document.createElement('button');
    saveBtn.className = 'save-btn';
    saveBtn.textContent = 'Save Settings';
    settingsGrid.appendChild(saveBtn);
}

// Initialize UI on load
initializeUI();

// ============================================================================
// CANVAS SETUP
// ============================================================================

const canvas = document.getElementById('cameraCanvas');
const ctx = canvas.getContext('2d');

/**
 * Draws grid pattern on canvas (placeholder for camera feed)
 */
function drawGrid() {
    const gridSize = 80;
    const rows = canvas.height / gridSize;
    const cols = canvas.width / gridSize;

    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#3a3a3a';
    ctx.lineWidth = 1;

    for (let i = 0; i <= cols; i++) {
        ctx.beginPath();
        ctx.moveTo(i * gridSize, 0);
        ctx.lineTo(i * gridSize, canvas.height);
        ctx.stroke();
    }

    for (let i = 0; i <= rows; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * gridSize);
        ctx.lineTo(canvas.width, i * gridSize);
        ctx.stroke();
    }
}

drawGrid();

// ============================================================================
// INPUT MANAGEMENT
// ============================================================================

// Get all number inputs (dynamically generated)
const inputs = SETTINGS_CONFIG.reduce((acc, config) => {
    acc[config.id] = document.getElementById(config.id);
    return acc;
}, {});

// ============================================================================
// SETTINGS CONTROLS
// ============================================================================

/**
 * Handles increment/decrement button clicks for number inputs
 */
document.querySelectorAll('.increment, .decrement').forEach(button => {
    button.addEventListener('click', () => {
        const targetId = button.getAttribute('data-target');
        const step = parseFloat(button.getAttribute('data-step'));
        const input = inputs[targetId];

        if (!input) return;

        const currentValue = parseFloat(input.value) || 0;
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);

        let newValue;
        if (button.classList.contains('increment')) {
            newValue = Math.min(currentValue + step, max);
        } else {
            newValue = Math.max(currentValue - step, min);
        }

        // Format based on step size
        if (step < 1) {
            input.value = newValue.toFixed(2);
        } else {
            input.value = Math.round(newValue);
        }
    });
});

/**
 * Restores all settings to default values
 */
const restoreBtn = document.getElementById('restoreDefaults');
restoreBtn.addEventListener('click', () => {
    Object.keys(DEFAULT_SETTINGS).forEach(key => {
        if (inputs[key]) {
            inputs[key].value = DEFAULT_SETTINGS[key];
        }
    });

    // Visual feedback
    restoreBtn.textContent = '✓ Restored!';
    restoreBtn.style.background = 'rgba(0, 255, 0, 0.2)';
    restoreBtn.style.borderColor = '#00ff00';
    restoreBtn.style.color = '#00ff00';

    setTimeout(() => {
        restoreBtn.textContent = '↻ Restore Defaults';
        restoreBtn.style.background = 'rgba(100, 100, 100, 0.3)';
        restoreBtn.style.borderColor = 'rgba(150, 150, 150, 0.3)';
        restoreBtn.style.color = '#ccc';
    }, 1500);
});

/**
 * Saves current settings to backend
 */
const saveBtn = document.querySelector('.save-btn');
saveBtn.addEventListener('click', async () => {
    // Get current settings from backend first
    try {
        const getResponse = await fetch(`${BACKEND_URL}/api/settings`);
        if (!getResponse.ok) {
            throw new Error('Failed to get current settings');
        }
        
        const currentData = await getResponse.json();
        const currentSettings = currentData.settings || {};
        
        // Merge with updated values - Map UI settings to backend config
        const updatedSettings = {
            ...currentSettings,
            health_monitoring: {
                ...currentSettings.health_monitoring,
                frame_rate: parseInt(inputs.frameRate?.value || 30),
                camera_index: parseInt(inputs.cameraIndex?.value || 0),
                DROWSY_THRESHOLD: parseFloat(inputs.drowsyThreshold?.value || 0.30),
                BLINK_THRESHOLD: parseFloat(inputs.blinkThreshold?.value || 0.27),
                MIN_REASONABLE_DISTANCE: parseInt(inputs.minReasonableDistance?.value || 30),
                MAX_REASONABLE_DISTANCE: parseInt(inputs.maxReasonableDistance?.value || 150)
            }
        };

        saveBtn.textContent = 'Saving...';
        saveBtn.disabled = true;

        const response = await fetch(`${BACKEND_URL}/api/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                settings: updatedSettings,
                reload_vision: true
            })
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Settings saved:', result);

            // Visual feedback
            saveBtn.textContent = result.reloaded ? 'Saved! Reloaded!' : 'Saved!';
            saveBtn.style.background = 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)';
            showNotification('Settings saved' + (result.reloaded ? ' and vision reloaded' : ''), 'success');

            setTimeout(() => {
                saveBtn.textContent = 'Save Settings';
                saveBtn.style.background = 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)';
                saveBtn.disabled = false;
            }, 2000);
        } else {
            const errorData = await response.json();
            throw new Error(errorData.message || 'Failed to save settings');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        saveBtn.textContent = 'Error!';
        saveBtn.style.background = 'linear-gradient(135deg, #ff4444 0%, #cc0000 100%)';
        showNotification('Failed to save settings: ' + error.message, 'error');

        setTimeout(() => {
            saveBtn.textContent = 'Save Settings';
            saveBtn.style.background = 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)';
            saveBtn.disabled = false;
        }, 2000);
    }
});

// ============================================================================
// WEBSOCKET CONNECTION & REAL-TIME DATA
// ============================================================================

// Backend server URL
const BACKEND_URL = 'http://localhost:5000';

// Initialize Socket.IO connection
const socket = io(BACKEND_URL);

// Connection status
socket.on('connect', () => {
    console.log('✅ Connected to AEyePro backend');
    showNotification('Connected to backend server', 'success');
});

socket.on('disconnect', () => {
    console.log('❌ Disconnected from backend');
    showNotification('Disconnected from backend', 'error');
});

socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    showNotification('Cannot connect to backend server', 'error');
});

socket.on('error', (data) => {
    console.error('Backend error:', data);
    showNotification(data.message || 'Backend error occurred', 'error');
});

/**
 * Handle real-time health metrics from backend
 */
socket.on('health_metrics', (data) => {
    updateHealthMetricsFromBackend(data);
});

/**
 * Handle camera frame data from backend
 */
socket.on('camera_frame', (data) => {
    updateCameraFrame(data.frame);
});

/**
 * Handle system status updates from backend
 */
socket.on('system_status', (status) => {
    console.log('System status:', status);
    updateSystemStatus(status);
});

/**
 * Update health metrics display with real backend data
 */
// Notification tracking để tránh spam
let lastDrowsyNotification = 0;
let lastBadPostureNotification = 0;
let lastDistanceWarning = 0;
const NOTIFICATION_COOLDOWN = 10000; // 10 giây

// Distance threshold for warning (will be loaded from settings)
window.minDistanceThreshold = 20; // Default 20cm

/**
 * Hiển thị pop-up notification
 */
function showPopupNotification(message, type = 'warning') {
    const now = Date.now();
    
    // Tạo notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'warning' ? '#ff6b6b' : '#ffa500'};
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-size: 16px;
        font-weight: bold;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        max-width: 350px;
    `;
    notification.textContent = message;
    
    // Thêm animation CSS nếu chưa có
    if (!document.getElementById('notificationStyles')) {
        const style = document.createElement('style');
        style.id = 'notificationStyles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(400px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(400px); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Tự động ẩn sau 5 giây
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function updateHealthMetricsFromBackend(data) {
    // Extract data from nested structure
    const eye = data.eye || {};
    const blink = data.blink || {};
    const posture = data.posture || {};
    const drowsiness = data.drowsiness || {};
    const system = data.system || {};

    // Update main metrics
    const blinkRate = document.getElementById('blinkRate');
    const avgEar = document.getElementById('avgEar');
    const avgDistance = document.getElementById('avgDistance');
    const drowsyEvents = document.getElementById('drowsyEvents');

    // Fix distance fallback: Try eye first, then posture
    const distanceCm = eye.distance_cm || posture.eye_distance_cm;

    if (blinkRate) blinkRate.textContent = blink.blink_rate ? `${blink.blink_rate.toFixed(1)} /min` : '-- /min';
    if (avgEar) avgEar.textContent = eye.avg_ear ? eye.avg_ear.toFixed(3) : '--';
    if (avgDistance) avgDistance.textContent = distanceCm ? `${distanceCm.toFixed(1)} cm` : '-- cm';
    
    // Update posture metrics
    const shoulderTilt = document.getElementById('shoulderTilt');
    const headPitch = document.getElementById('headPitch');
    const headYaw = document.getElementById('headYaw');
    
    if (shoulderTilt) shoulderTilt.textContent = posture.shoulder_tilt != null ? `${posture.shoulder_tilt.toFixed(1)}°` : '--°';
    if (headPitch) headPitch.textContent = posture.head_updown_angle != null ? `${posture.head_updown_angle.toFixed(1)}°` : '--°';
    if (headYaw) headYaw.textContent = posture.head_side_angle != null ? `${posture.head_side_angle.toFixed(1)}°` : '--°';
    
    // Track drowsiness events (would need to accumulate)
    if (drowsyEvents) {
        if (drowsiness.detected) {
            const currentCount = parseInt(drowsyEvents.textContent) || 0;
            // Only increment if this is a new detection
            if (!window.lastDrowsyState) {
                drowsyEvents.textContent = currentCount + 1;
            }
            window.lastDrowsyState = true;
        } else {
            window.lastDrowsyState = false;
        }
    }
    
    // POP-UP NOTIFICATIONS cho drowsiness, bad posture, và distance warning
    const now = Date.now();
    
    // DEBUG: Log notification checks
    if (Math.random() < 0.05) {  // Log 5% of checks
        console.log('[NOTIFICATION CHECK]', {
            drowsy_detected: drowsiness.detected,
            posture_status: posture.status,
            distance_cm: distanceCm,
            cooldown_drowsy: (now - lastDrowsyNotification) / 1000 + 's',
            cooldown_posture: (now - lastBadPostureNotification) / 1000 + 's'
        });
    }
    
    // Check distance warning - if face too close to screen
    if (distanceCm) {
        const tooClose = distanceCm < window.minDistanceThreshold + 1;
        const cooldownPassed = (now - lastDistanceWarning) > NOTIFICATION_COOLDOWN;
        
        if (tooClose && cooldownPassed) {
            console.log('[NOTIFICATION] ⚠️ Showing distance warning!');
            showPopupNotification('⚠️ TOO CLOSE! Move back from the screen!', 'warning');
            lastDistanceWarning = now;
        }
    }
    
    // Check drowsiness - hiện notification
    if (drowsiness.detected && (now - lastDrowsyNotification) > NOTIFICATION_COOLDOWN) {
        console.log('[NOTIFICATION] ⚠️ Showing drowsiness alert!');
        showPopupNotification('⚠️ DROWSINESS DETECTED! Take a break!', 'warning');
        lastDrowsyNotification = now;
    }
    
    // Check bad posture - hiện notification
    if (posture.status === 'poor' && (now - lastBadPostureNotification) > NOTIFICATION_COOLDOWN) {
        console.log('[NOTIFICATION] ⚠️ Showing bad posture alert!');
        showPopupNotification('⚠️ BAD POSTURE! Adjust your position!', 'warning');
        lastBadPostureNotification = now;
    }
}

/**
 * Update camera canvas with real video frame (base64 encoded)
 */
function updateCameraFrame(base64Frame) {
    if (!base64Frame) return;

    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.onerror = () => {
        console.error('Failed to load camera frame');
    };
    img.src = 'data:image/jpeg;base64,' + base64Frame;
}

/**
 * Update system status indicators
 */
function updateSystemStatus(status) {
    // Update camera state based on backend status
    if (status.is_running !== undefined) {
        cameraOn = status.is_running;
        
        if (cameraOn) {
            cameraToggleBtn.classList.remove('off');
            cameraToggleBtn.querySelector('span').textContent = 'Turn Off';
            liveBadge.style.display = 'flex';
        } else {
            cameraToggleBtn.classList.add('off');
            cameraToggleBtn.querySelector('span').textContent = 'Turn On';
            liveBadge.style.display = 'none';
        }
    }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add some basic styling
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.borderRadius = '8px';
    notification.style.zIndex = '10000';
    notification.style.transition = 'opacity 0.3s';
    notification.style.fontFamily = 'Orbitron, sans-serif';
    notification.style.fontSize = '14px';
    notification.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
    
    // Type-specific styling
    if (type === 'success') {
        notification.style.background = 'rgba(0, 255, 136, 0.2)';
        notification.style.border = '1px solid #00ff88';
        notification.style.color = '#00ff88';
    } else if (type === 'error') {
        notification.style.background = 'rgba(255, 68, 68, 0.2)';
        notification.style.border = '1px solid #ff4444';
        notification.style.color = '#ff4444';
    } else {
        notification.style.background = 'rgba(0, 212, 255, 0.2)';
        notification.style.border = '1px solid #00d4ff';
        notification.style.color = '#00d4ff';
    }
    
    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================================
// CHAT FUNCTIONALITY
// ============================================================================

const chatInput = document.querySelector('.chat-input');
const sendBtn = document.querySelector('.send-btn');
const chatWindow = document.getElementById('chatWindow');

/**
 * Creates a message element
 * @param {string} message - The message text
 * @param {boolean} isUser - Whether this is a user message
 * @returns {HTMLElement} Message element
 */
function createMessageElement(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = message; // Use textContent to prevent XSS

    messageDiv.appendChild(bubble);
    return messageDiv;
}

/**
 * Sends a chat message
 */
function sendMessage() {
    const message = chatInput.value.trim();
    if (message === '') return;

    // Add user message
    chatWindow.appendChild(createMessageElement(message, true));

    // Clear input
    chatInput.value = '';

    // Scroll to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Simulate bot response
    setTimeout(() => {
        const botResponse = `Processing your request: "${message}"`;
        chatWindow.appendChild(createMessageElement(botResponse, false));
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }, 1000);
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// ============================================================================
// WINDOW CONTROLS
// ============================================================================

// Title bar buttons
document.querySelector('.title-btn.minimize').addEventListener('click', () => {
    console.log('Minimize window');
});

document.querySelector('.title-btn.maximize').addEventListener('click', () => {
    console.log('Maximize window');
});

document.querySelector('.title-btn.close').addEventListener('click', () => {
    console.log('Close window');
});

// ============================================================================
// CAMERA CONTROLS
// ============================================================================

/**
 * Camera toggle state and controls
 */
let cameraOn = true;
const cameraToggleBtn = document.getElementById('cameraToggle');
const liveBadge = document.querySelector('.live-badge');

/**
 * Face Mesh toggle state and controls
 */
let faceMeshEnabled = true;
const faceMeshToggleBtn = document.getElementById('faceMeshToggle');

faceMeshToggleBtn.addEventListener('click', async () => {
    faceMeshEnabled = !faceMeshEnabled;
    
    // Update button appearance
    if (faceMeshEnabled) {
        faceMeshToggleBtn.classList.add('active');
    } else {
        faceMeshToggleBtn.classList.remove('active');
    }
    
    // Send toggle state to backend
    try {
        const response = await fetch(`${BACKEND_URL}/api/settings/face-mesh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: faceMeshEnabled })
        });
        
        if (response.ok) {
            console.log('Face mesh toggled:', faceMeshEnabled);
        }
    } catch (error) {
        console.error('Failed to toggle face mesh:', error);
    }
});

cameraToggleBtn.addEventListener('click', async () => {
    const targetState = !cameraOn;

    try {
        cameraToggleBtn.disabled = true;
        const endpoint = targetState ? '/api/camera/start' : '/api/camera/stop';

        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            method: 'POST'
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Camera toggle:', result);

            cameraOn = targetState;

            if (cameraOn) {
                // Turn camera ON
                cameraToggleBtn.classList.remove('off');
                cameraToggleBtn.querySelector('span').textContent = 'Turn Off';
                liveBadge.style.display = 'flex';
                showNotification('Camera started', 'success');
            } else {
                // Turn camera OFF
                cameraToggleBtn.classList.add('off');
                cameraToggleBtn.querySelector('span').textContent = 'Turn On';
                liveBadge.style.display = 'none';
                // Clear canvas
                ctx.fillStyle = '#1a1a1a';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                showNotification('Camera stopped', 'info');
            }
        } else {
            throw new Error('Failed to toggle camera');
        }
    } catch (error) {
        console.error('Error toggling camera:', error);
        showNotification('Failed to toggle camera', 'error');
    } finally {
        cameraToggleBtn.disabled = false;
    }
});

// Camera feed simulation
function simulateCameraFeed() {
    const liveIndicator = document.querySelector('.pulse-dot');
    setInterval(() => {
        if (cameraOn) {
            liveIndicator.style.opacity = liveIndicator.style.opacity === '0.5' ? '1' : '0.5';
        }
    }, 750);
}

simulateCameraFeed();

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the application and load settings
 */
async function initializeApp() {
    try {
        console.log('Initializing AEyePro UI...');

        // Check backend status
        const statusResponse = await fetch(`${BACKEND_URL}/api/camera/status`);
        if (statusResponse.ok) {
            const status = await statusResponse.json();
            console.log('Backend status:', status);
            updateSystemStatus(status);
            
            // If camera is already running, update UI
            if (status.is_running) {
                cameraOn = true;
                cameraToggleBtn.classList.remove('off');
                cameraToggleBtn.querySelector('span').textContent = 'Turn Off';
                liveBadge.style.display = 'flex';
            }
        }
        
        // Initialize face mesh toggle button state
        if (faceMeshEnabled) {
            faceMeshToggleBtn.classList.add('active');
        }

        // Load current settings from backend
        const settingsResponse = await fetch(`${BACKEND_URL}/api/settings`);
        if (settingsResponse.ok) {
            const settingsData = await settingsResponse.json();
            if (settingsData.success && settingsData.settings) {
                const healthMonitoring = settingsData.settings.health_monitoring || {};
                
                // Populate settings UI with current values (6 simplified controls)
                if (inputs.frameRate && healthMonitoring.frame_rate !== undefined) {
                    inputs.frameRate.value = healthMonitoring.frame_rate;
                }
                if (inputs.cameraIndex && healthMonitoring.camera_index !== undefined) {
                    inputs.cameraIndex.value = healthMonitoring.camera_index;
                }
                if (inputs.drowsyThreshold && healthMonitoring.DROWSY_THRESHOLD !== undefined) {
                    inputs.drowsyThreshold.value = healthMonitoring.DROWSY_THRESHOLD;
                }
                if (inputs.blinkThreshold && healthMonitoring.BLINK_THRESHOLD !== undefined) {
                    inputs.blinkThreshold.value = healthMonitoring.BLINK_THRESHOLD;
                }
                if (inputs.minReasonableDistance && healthMonitoring.MIN_REASONABLE_DISTANCE !== undefined) {
                    inputs.minReasonableDistance.value = healthMonitoring.MIN_REASONABLE_DISTANCE;
                    window.minDistanceThreshold = healthMonitoring.MIN_REASONABLE_DISTANCE; // Store for distance warning
                }
                if (inputs.maxReasonableDistance && healthMonitoring.MAX_REASONABLE_DISTANCE !== undefined) {
                    inputs.maxReasonableDistance.value = healthMonitoring.MAX_REASONABLE_DISTANCE;
                }
                
                console.log('Settings loaded from backend');
            }
        }

        showNotification('AEyePro UI initialized', 'success');
    } catch (error) {
        console.error('Failed to initialize:', error);
        showNotification('Backend server not available. Please start backend_server.py', 'error');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

console.log('AeyePro Vision Module UI initialized successfully');
