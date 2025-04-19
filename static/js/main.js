async function resetStatistics(type, cameraId = null) {
    try {
        const params = new URLSearchParams();
        params.append('reset_type', type);
        if (cameraId) {
            params.append('camera_id', cameraId);
        }
        
        const response = await fetch(`/api/stats/reset?${params.toString()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Success', `Reset ${type} statistics successfully`, 'success');
            updateStatsSummary();  // Refresh the statistics display
        } else {
            showNotification('Error', data.detail || 'Failed to reset statistics', 'error');
        }
    } catch (error) {
        console.error('Error resetting statistics:', error);
        showNotification('Error', 'Failed to reset statistics', 'error');
    }
}

async function updateStatsSummary() {
    try {
        const response = await fetch('/api/stats/summary');
        const data = await response.json();
        
        if (response.ok) {
            // Update statistics display
            const stats = data.data;
            updateStatsDisplay(stats);
        } else {
            console.error('Failed to fetch statistics:', data.detail);
        }
    } catch (error) {
        console.error('Error updating statistics:', error);
    }
}

function updateStatsDisplay(stats) {
    // Update performance metrics
    if (stats.performance_metrics) {
        document.getElementById('performance-count').textContent = stats.performance_metrics.count;
        document.getElementById('performance-oldest').textContent = formatDate(stats.performance_metrics.oldest);
        document.getElementById('performance-newest').textContent = formatDate(stats.performance_metrics.newest);
    }
    
    // Update face detections
    if (stats.face_detections) {
        document.getElementById('faces-count').textContent = stats.face_detections.count;
        document.getElementById('faces-oldest').textContent = formatDate(stats.face_detections.oldest);
        document.getElementById('faces-newest').textContent = formatDate(stats.face_detections.newest);
    }
    
    // Update error logs
    if (stats.error_logs) {
        document.getElementById('errors-count').textContent = stats.error_logs.count;
        document.getElementById('errors-oldest').textContent = formatDate(stats.error_logs.oldest);
        document.getElementById('errors-newest').textContent = formatDate(stats.error_logs.newest);
    }
    
    // Update storage usage
    if (stats.storage) {
        const sizeInMB = (stats.storage.total_size / (1024 * 1024)).toFixed(2);
        document.getElementById('storage-size').textContent = `${sizeInMB} MB`;
    }
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
}

function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <h4>${title}</h4>
        <p>${message}</p>
    `;
    
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Initialize statistics on page load
document.addEventListener('DOMContentLoaded', () => {
    updateStatsSummary();
}); 