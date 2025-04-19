// Camera management functionality
let cameras = [];
const cameraGrid = document.getElementById('camera-grid');
const addCameraModal = new bootstrap.Modal(document.getElementById('addCameraModal'));
const editCameraModal = new bootstrap.Modal(document.getElementById('editCameraModal'));

// Fetch cameras and update the grid
async function fetchCameras() {
    try {
        const response = await fetch('/api/cameras');
        if (!response.ok) throw new Error('Failed to fetch cameras');
        cameras = await response.json();
        updateCameraGrid();
    } catch (error) {
        console.error('Error fetching cameras:', error);
        showAlert('Error fetching cameras', 'danger');
    }
}

// Update the camera grid with current camera data
function updateCameraGrid() {
    cameraGrid.innerHTML = '';
    cameras.forEach(camera => {
        const card = createCameraCard(camera);
        cameraGrid.appendChild(card);
    });
}

// Create a camera card element
function createCameraCard(camera) {
    const card = document.createElement('div');
    card.className = 'col-md-6 col-lg-4 mb-4';
    card.innerHTML = `
        <div class="card h-100">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">${camera.name || 'Unnamed Camera'}</h5>
                <div class="btn-group">
                    <button class="btn btn-sm btn-primary" onclick="toggleCamera(${camera.id})">
                        <i class="fas fa-${camera.running ? 'stop' : 'play'}"></i>
                    </button>
                    <button class="btn btn-sm btn-secondary" onclick="editCamera(${camera.id})">
                        <i class="fas fa-cog"></i>
                    </button>
                </div>
            </div>
            <div class="card-body text-center">
                <img src="/api/cameras/${camera.id}/snapshot" 
                     class="img-fluid camera-feed" 
                     alt="Camera Feed"
                     onerror="this.src='/static/img/offline.png'">
            </div>
            <div class="card-footer">
                <small class="text-muted">
                    Status: ${camera.running ? 'Running' : 'Stopped'}<br>
                    Resolution: ${camera.width}x${camera.height}<br>
                    FPS: ${camera.fps}
                </small>
            </div>
        </div>
    `;
    return card;
}

// Toggle camera state (start/stop)
async function toggleCamera(cameraId) {
    try {
        const camera = cameras.find(c => c.id === cameraId);
        if (!camera) throw new Error('Camera not found');

        const action = camera.running ? 'stop' : 'start';
        const response = await fetch(`/api/cameras/${cameraId}/${action}`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error(`Failed to ${action} camera`);
        await fetchCameras();
    } catch (error) {
        console.error(`Error toggling camera ${cameraId}:`, error);
        showAlert(`Error toggling camera: ${error.message}`, 'danger');
    }
}

// Edit camera settings
function editCamera(cameraId) {
    const camera = cameras.find(c => c.id === cameraId);
    if (!camera) {
        console.error('Camera not found:', cameraId);
        return;
    }

    // Populate edit form
    document.getElementById('editCameraId').value = camera.id;
    document.getElementById('editCameraName').value = camera.name || '';
    document.getElementById('editCameraSource').value = camera.source || '';
    document.getElementById('editCameraWidth').value = camera.width || '';
    document.getElementById('editCameraHeight').value = camera.height || '';
    document.getElementById('editCameraFps').value = camera.fps || '';
    document.getElementById('editCameraEnabled').checked = camera.enabled || false;

    editCameraModal.show();
}

// Save camera settings
async function saveCameraSettings(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const cameraId = formData.get('id');

    const settings = {
        name: formData.get('name'),
        source: formData.get('source'),
        width: parseInt(formData.get('width')),
        height: parseInt(formData.get('height')),
        fps: parseInt(formData.get('fps')),
        enabled: formData.get('enabled') === 'on'
    };

    try {
        const response = await fetch(`/api/cameras/${cameraId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (!response.ok) throw new Error('Failed to update camera settings');
        
        editCameraModal.hide();
        await fetchCameras();
        showAlert('Camera settings updated successfully', 'success');
    } catch (error) {
        console.error('Error saving camera settings:', error);
        showAlert(`Error saving camera settings: ${error.message}`, 'danger');
    }
}

// Add new camera
async function addCamera(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    const newCamera = {
        name: formData.get('name'),
        source: formData.get('source'),
        width: parseInt(formData.get('width')),
        height: parseInt(formData.get('height')),
        fps: parseInt(formData.get('fps')),
        enabled: formData.get('enabled') === 'on'
    };

    try {
        const response = await fetch('/api/cameras', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newCamera)
        });

        if (!response.ok) throw new Error('Failed to add camera');
        
        addCameraModal.hide();
        event.target.reset();
        await fetchCameras();
        showAlert('Camera added successfully', 'success');
    } catch (error) {
        console.error('Error adding camera:', error);
        showAlert(`Error adding camera: ${error.message}`, 'danger');
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    alertsContainer.appendChild(alert);
    setTimeout(() => alert.remove(), 5000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    fetchCameras();
    setInterval(fetchCameras, 5000); // Refresh every 5 seconds

    // Set up form submit handlers
    document.getElementById('addCameraForm').addEventListener('submit', addCamera);
    document.getElementById('editCameraForm').addEventListener('submit', saveCameraSettings);
}); 