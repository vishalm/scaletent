/**
 * ScaleTent Main JavaScript
 * Common functionality for all pages
 */

// Global configuration
const config = {
    apiBaseUrl: 'http://localhost:5000/api',
    wsBaseUrl: window.location.protocol === 'https:' ? 'wss://' : 'ws://' + window.location.host + '/api/ws',
    updateInterval: 5000
};

// Utility functions
const utils = {
    formatDate: (date) => {
        return new Date(date).toLocaleString();
    },
    
    formatNumber: (num) => {
        return new Intl.NumberFormat().format(num);
    },
    
    showToast: (message, type = 'success') => {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
    },
    
    handleApiError: (error) => {
        console.error('API Error:', error);
        utils.showToast('An error occurred. Please try again.', 'danger');
    }
};

// API functions
const api = {
    async get(endpoint) {
        try {
            const response = await fetch(`${config.apiBaseUrl}${endpoint}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            utils.handleApiError(error);
            throw error;
        }
    },
    
    async post(endpoint, data) {
        try {
            const response = await fetch(`${config.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            utils.handleApiError(error);
            throw error;
        }
    },
    
    async put(endpoint, data) {
        try {
            const response = await fetch(`${config.apiBaseUrl}${endpoint}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            utils.handleApiError(error);
            throw error;
        }
    },
    
    async delete(endpoint) {
        try {
            const response = await fetch(`${config.apiBaseUrl}${endpoint}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            utils.handleApiError(error);
            throw error;
        }
    }
};

// WebSocket connection manager
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.handlers = new Map();
    }
    
    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) return;
        
        this.ws = new WebSocket(config.wsBaseUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type && this.handlers.has(data.type)) {
                    this.handlers.get(data.type)(data);
                }
            } catch (error) {
                console.error('WebSocket message error:', error);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    addHandler(type, handler) {
        this.handlers.set(type, handler);
    }
    
    removeHandler(type) {
        this.handlers.delete(type);
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Initialize WebSocket manager
const wsManager = new WebSocketManager();

// Export utilities and managers
window.scaletent = {
    config,
    utils,
    api,
    wsManager
}; 