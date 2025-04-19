/**
 * ScaleTent Dashboard JavaScript
 * Handles real-time updates and interaction with the dashboard UI
 */

// WebSocket connection
let ws;
let reconnectInterval = 1000; // Start with 1s reconnect interval
const maxReconnectInterval = 30000; // Max reconnect interval (30s)
let reconnectTimer = null;
let isConnected = false;

// Chart references
let recognitionChart;

// System status update timer
let statusTimer = null;

/**
 * Initialize the dashboard
 */
function initDashboard() {
    // Initialize counters
    updateCounters(0, 0, 0);
    
    // Initialize system status
    updateSystemStatus();
    
    // Start status update timer (every 10 seconds)
    statusTimer = setInterval(updateSystemStatus, 10000);
}

/**
 * Connect to the WebSocket server
 */
function connectWebSocket() {
    // Clear any existing reconnect timer
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    
    // Get the websocket URL (ws:// or wss:// depending on the current protocol)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/detection_stream`;
    
    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    // Create WebSocket connection
    ws = new WebSocket(wsUrl);
    
    // Connection opened
    ws.addEventListener('open', (event) => {
        console.log('Connected to WebSocket server');
        isConnected = true;
        reconnectInterval = 1000; // Reset reconnect interval
        
        // Update UI to show connected status
        $('#system-status-value').text('Running').removeClass('text-danger').addClass('text-success');
    });
    
    // Listen for messages
    ws.addEventListener('message', (event) => {
        // Process the received message
        processWebSocketMessage(event.data);
    });
    
    // Connection closed
    ws.addEventListener('close', (event) => {
        console.log('Disconnected from WebSocket server');
        isConnected = false;
        
        // Update UI to show disconnected status
        $('#system-status-value').text('Disconnected').removeClass('text-success').addClass('text-danger');
        
        // Schedule reconnect
        scheduleReconnect();
    });
    
    // Connection error
    ws.addEventListener('error', (event) => {
        console.error('WebSocket error:', event);
        
        // Update UI to show error status
        $('#system-status-value').text('Error').removeClass('text-success').addClass('text-danger');
        
        // The connection will close after an error
    });
}

/**
 * Schedule a reconnect attempt
 */
function scheduleReconnect() {
    console.log(`Scheduling reconnect in ${reconnectInterval}ms`);
    
    reconnectTimer = setTimeout(() => {
        console.log('Attempting to reconnect...');
        connectWebSocket();
    }, reconnectInterval);
    
    // Exponential backoff for reconnect interval
    reconnectInterval = Math.min(reconnectInterval * 1.5, maxReconnectInterval);
}

/**
 * Process a WebSocket message
 * @param {string} data - JSON message from WebSocket
 */
function processWebSocketMessage(data) {
    try {
        // Parse JSON data
        const message = JSON.parse(data);
        
        // Update live feed (if from selected camera)
        updateLiveFeed(message);
        
        // Update counters
        const peopleCount = message.detections ? message.detections.length : 0;
        const recognizedCount = message.detections ? message.detections.filter(d => d.recognized).length : 0;
        const unknownCount = peopleCount - recognizedCount;
        
        updateCounters(peopleCount, recognizedCount, unknownCount);
        
        // Add to latest detections
        updateLatestDetections(message);
        
        // Update chart data
        updateChartData(message);
        
    } catch (error) {
        console.error('Error processing WebSocket message:', error);
    }
}

/**
 * Update the live feed and detection overlay
 * @param {object} message - Detection message
 */
function updateLiveFeed(message) {
    // Check if this message is from the currently selected camera
    const selectedCamera = $('#camera-select').val();
    
    if (message.camera_id === selectedCamera) {
        // In a real implementation, this would update the live feed image
        // For now, we'll just update the detection overlay
        
        // Clear existing overlay
        const overlay = $('#detection-overlay');
        overlay.empty();
        
        // Add detection boxes
        if (message.detections && message.detections.length > 0) {
            message.detections.forEach(detection => {
                addDetectionBox(overlay, detection);
            });
        }
    }
}

/**
 * Add a detection box to the overlay
 * @param {jQuery} overlay - The overlay container
 * @param {object} detection - Detection data
 */
function addDetectionBox(overlay, detection) {
    // Get normalized coordinates
    const [x1, y1, x2, y2] = detection.bbox;
    
    // Create detection box
    const box = $('<div class="detection-box"></div>');
    
    // Set position and size (assumes image is 100% of container)
    const feedContainer = $('#live-feed-image').parent();
    const containerWidth = feedContainer.width();
    const containerHeight = feedContainer.height();
    
    // Calculate positions in container
    // This is a simplification - in a real implementation, you'd need to
    // handle the aspect ratio and scaling of the feed image
    const feedWidth = 640; // Assumed feed width
    const feedHeight = 480; // Assumed feed height
    
    const boxX = (x1 / feedWidth) * containerWidth;
    const boxY = (y1 / feedHeight) * containerHeight;
    const boxWidth = ((x2 - x1) / feedWidth) * containerWidth;
    const boxHeight = ((y2 - y1) / feedHeight) * containerHeight;
    
    box.css({
        left: boxX + 'px',
        top: boxY + 'px',
        width: boxWidth + 'px',
        height: boxHeight + 'px',
        borderColor: detection.recognized ? '#4caf50' : '#ff9800'
    });
    
    // Add label
    const label = $('<div class="detection-label"></div>');
    if (detection.recognized && detection.person_data) {
        label.text(detection.person_data.name || detection.person_data.id);
        label.addClass('recognized');
    } else {
        label.text('Unknown');
        label.addClass('unknown');
    }
    
    box.append(label);
    overlay.append(box);
}

/**
 * Update the counter displays
 * @param {number} peopleCount - Number of people detected
 * @param {number} recognizedCount - Number of recognized people
 * @param {number} unknownCount - Number of unknown people
 */
function updateCounters(peopleCount, recognizedCount, unknownCount) {
    $('#people-count').text(peopleCount);
    $('#recognized-count').text(recognizedCount);
    $('#unknown-count').text(unknownCount);
    
    // Camera count is updated separately via the system status
}

/**
 * Update the latest detections panel
 * @param {object} message - Detection message
 */
function updateLatestDetections(message) {
    const container = $('#latest-detections');
    
    // If this is the first detection, clear the waiting message
    if (container.find('.alert-info').length > 0) {
        container.empty();
    }
    
    // Keep only the latest 10 detections
    if (container.children().length >= 10) {
        container.children().last().remove();
    }
    
    // Process each detection
    if (message.detections && message.detections.length > 0) {
        message.detections.forEach(detection => {
            // Skip if not a person
            if (detection.type !== 'person') return;
            
            // Create detection entry
            const entry = $('<div class="detection-entry"></div>');
            
            // Add timestamp
            const timestamp = new Date(message.timestamp);
            const timeStr = timestamp.toLocaleTimeString();
            
            // Add content based on recognition status
            if (detection.recognized && detection.person_data) {
                entry.html(`
                    <div class="detection-time">${timeStr}</div>
                    <div class="detection-info recognized">
                        <i class="fas fa-user-check"></i>
                        <strong>${detection.person_data.name || detection.person_data.id}</strong>
                        ${detection.person_data.role ? `<span>(${detection.person_data.role})</span>` : ''}
                    </div>
                    <div class="detection-camera">${message.camera_id}</div>
                `);
            } else {
                entry.html(`
                    <div class="detection-time">${timeStr}</div>
                    <div class="detection-info unknown">
                        <i class="fas fa-user-times"></i>
                        <strong>Unknown Person</strong>
                    </div>
                    <div class="detection-camera">${message.camera_id}</div>
                `);
            }
            
            // Add to container at the top
            container.prepend(entry);
        });
    }
}

/**
 * Initialize the recognition timeline chart
 */
function initRecognitionChart() {
    const ctx = document.getElementById('recognitionChart').getContext('2d');
    
    // Create 24 hour labels (hourly)
    const labels = [];
    for (let i = 0; i < 24; i++) {
        labels.push(`${i}:00`);
    }
    
    // Initialize with zero data
    const recognizedData = new Array(24).fill(0);
    const unknownData = new Array(24).fill(0);
    
    recognitionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Recognized',
                    data: recognizedData,
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Unknown',
                    data: unknownData,
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 2,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}

/**
 * Update chart data based on detection message
 * @param {object} message - Detection message
 */
function updateChartData(message) {
    // Get current hour
    const timestamp = new Date(message.timestamp);
    const hour = timestamp.getHours();
    
    // Update chart data for the current hour
    if (message.detections && message.detections.length > 0) {
        // Count recognized and unknown persons
        const recognizedCount = message.detections.filter(d => d.recognized).length;
        const unknownCount = message.detections.length - recognizedCount;
        
        // Update chart data
        recognitionChart.data.datasets[0].data[hour] += recognizedCount;
        recognitionChart.data.datasets[1].data[hour] += unknownCount;
        
        // Update chart
        recognitionChart.update();
    }
}

/**
 * Update system status information
 */
function updateSystemStatus() {
    // Make API request to get system status
    $.ajax({
        url: '/api/v1/status',
        method: 'GET',
        dataType: 'json',
        success: function(data) {
            // Update uptime
            const uptime = formatUptime(data.uptime);
            $('#uptime').text(uptime);
            
            // Update detection FPS
            $('#detection-fps').text(data.detection_fps.toFixed(1));
            
            // Update connected clients
            $('#connected-clients').text(data.connected_clients);
            
            // Update camera count
            const cameraCount = Object.keys(data.cameras).length;
            $('#camera-count').text(cameraCount);
            
            // Update camera select options if needed
            updateCameraOptions(data.cameras);
            
            // Update system status text
            $('#system-status-value').text(data.status).removeClass('text-danger').addClass('text-success');
        },
        error: function(xhr, status, error) {
            console.error('Error fetching system status:', error);
            $('#system-status-value').text('Error').removeClass('text-success').addClass('text-danger');
        }
    });
}

/**
 * Update camera selection options
 * @param {object} cameras - Camera information
 */
function updateCameraOptions(cameras) {
    const select = $('#camera-select');
    const currentValue = select.val();
    
    // Clear existing options
    select.empty();
    
    // Add options for each camera
    Object.keys(cameras).forEach(cameraId => {
        const camera = cameras[cameraId];
        select.append(`<option value="${cameraId}">${camera.name || cameraId}</option>`);
    });
    
    // Restore previous selection if it still exists
    if (currentValue && select.find(`option[value="${currentValue}"]`).length > 0) {
        select.val(currentValue);
    }
}

/**
 * Format uptime as HH:MM:SS
 * @param {number} seconds - Uptime in seconds
 * @returns {string} Formatted uptime
 */
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return [
        hours.toString().padStart(2, '0'),
        minutes.toString().padStart(2, '0'),
        secs.toString().padStart(2, '0')
    ].join(':');
}

/**
 * Change the selected camera
 * @param {string} cameraId - Camera ID
 */
function changeCamera(cameraId) {
    // In a real implementation, this would update the live feed source
    console.log(`Switching to camera: ${cameraId}`);
    
    // Clear detection overlay
    $('#detection-overlay').empty();
}

/**
 * Submit the Add Person form
 */
function submitAddPersonForm() {
    // Get form values
    const personId = $('#person-id').val();
    const personName = $('#person-name').val();
    const personRole = $('#person-role').val();
    const personOrg = $('#person-organization').val();
    const personImage = $('#person-image')[0].files[0];
    
    // Validate required fields
    if (!personId || !personName || !personImage) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('person_id', personId);
    formData.append('name', personName);
    
    if (personRole) {
        formData.append('role', personRole);
    }
    
    if (personOrg) {
        formData.append('organization', personOrg);
    }
    
    formData.append('image', personImage);
    
    // Send API request
    $.ajax({
        url: '/api/v1/upload_profile',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            // Show success notification
            alert('Person added successfully!');
            
            // Close modal
            $('#addPersonModal').modal('hide');
            
            // Clear form
            $('#add-person-form')[0].reset();
        },
        error: function(xhr, status, error) {
            // Show error notification
            alert('Error adding person: ' + xhr.responseJSON?.detail || error);
        }
    });
}

/**
 * Export data based on form values
 */
function exportData() {
    // Get form values
    const dataType = $('#export-type').val();
    const format = $('#export-format').val();
    const startDate = $('#export-start-date').val();
    const endDate = $('#export-end-date').val();
    
    // Validate dates
    if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
        alert('Start date must be before end date');
        return;
    }
    
    // Build request URL
    let url = `/api/v1/export/${dataType}?format=${format}`;
    
    if (startDate) {
        url += `&start_date=${startDate}`;
    }
    
    if (endDate) {
        url += `&end_date=${endDate}`;
    }
    
    // Initiate download
    window.location.href = url;
    
    // Close modal
    $('#exportDataModal').modal('hide');
}

/**
 * Clean up resources when page is unloaded
 */
window.addEventListener('beforeunload', function() {
    // Close WebSocket connection
    if (ws) {
        ws.close();
    }
    
    // Clear timers
    if (statusTimer) {
        clearInterval(statusTimer);
    }
    
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
    }
});

function updateCameraOptions(cameras) {
    const select = $('#camera-select');
    const currentValue = select.val();
    
    // Clear existing options
    select.empty();
    
    // Add options for each camera
    Object.keys(cameras).forEach(cameraId => {
        const camera = cameras[cameraId];
        select.append(`<option value="${cameraId}">${camera.name || cameraId}</option>`);
    });
    
    // Restore previous selection if it still exists
    if (currentValue && select.find(`option[value="${currentValue}"]`).length > 0) {
        select.val(currentValue);
    }
}

/**
 * Format uptime as HH:MM:SS
 * @param {number} seconds - Uptime in seconds
 * @returns {string} Formatted uptime
 */
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return [
        hours.toString().padStart(2, '0'),
        minutes.toString().padStart(2, '0'),
        secs.toString().padStart(2, '0')
    ].join(':');
}

/**
 * Change the selected camera
 * @param {string} cameraId - Camera ID
 */
function changeCamera(cameraId) {
    // In a real implementation, this would update the live feed source
    console.log(`Switching to camera: ${cameraId}`);
    
    // Clear detection overlay
    $('#detection-overlay').empty();
}

/**
 * Submit the Add Person form
 */
function submitAddPersonForm() {
    // Get form values
    const personId = $('#person-id').val();
    const personName = $('#person-name').val();
    const personRole = $('#person-role').val();
    const personOrg = $('#person-organization').val();
    const personImage = $('#person-image')[0].files[0];
    
    // Validate required fields
    if (!personId || !personName || !personImage) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('person_id', personId);
    formData.append('name', personName);
    
    if (personRole) {
        formData.append('role', personRole);
    }
    
    if (personOrg) {
        formData.append('organization', personOrg);
    }
    
    formData.append('image', personImage);
    
    //