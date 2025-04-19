-- Configuration tables

CREATE TABLE IF NOT EXISTS config_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS config_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT NOT NULL,  -- 'str', 'int', 'float', 'bool', 'json'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (section_id) REFERENCES config_sections(id),
    UNIQUE(section_id, key)
);

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT UNIQUE NOT NULL,
    name TEXT,
    source TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps INTEGER NOT NULL,
    processing_fps INTEGER DEFAULT 15,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Metrics tables
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    fps FLOAT,
    processing_time FLOAT,
    queue_size INTEGER,
    faces_detected INTEGER,
    faces_recognized INTEGER,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS recognition_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    person_id TEXT,
    confidence_score FLOAT,
    detection_time FLOAT,
    recognition_time FLOAT,
    embedding_generation_time FLOAT,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

-- Recording Management
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    file_path TEXT NOT NULL,
    file_size INTEGER,  -- in bytes
    duration INTEGER,   -- in seconds
    format TEXT,
    quality TEXT,
    status TEXT,       -- 'recording', 'completed', 'failed'
    error_message TEXT,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS recording_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL,
    segment_number INTEGER NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    file_path TEXT NOT NULL,
    file_size INTEGER,  -- in bytes
    status TEXT,       -- 'recording', 'completed', 'failed'
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- Event Tracking
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'face_detected', 'person_recognized', 'error', 'performance_issue'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT,  -- JSON formatted details
    severity TEXT, -- 'info', 'warning', 'error'
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

-- Face Detection Storage
CREATE TABLE IF NOT EXISTS face_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_id TEXT,
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    confidence FLOAT,
    person_id TEXT,    -- NULL if not recognized
    embedding BLOB,    -- stored as binary
    thumbnail_path TEXT,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

-- System Health Monitoring
CREATE TABLE IF NOT EXISTS system_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_usage FLOAT,
    temperature FLOAT,
    uptime INTEGER,
    active_cameras INTEGER,
    total_fps FLOAT,
    error_count INTEGER
);

-- Error Logging
CREATE TABLE IF NOT EXISTS error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_type TEXT,
    error_message TEXT,
    stack_trace TEXT,
    camera_id TEXT,    -- NULL if not camera-specific
    severity TEXT,     -- 'warning', 'error', 'critical'
    resolved BOOLEAN DEFAULT 0,
    resolution_time TIMESTAMP,
    resolution_notes TEXT,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

-- Cleanup and Maintenance
CREATE TABLE IF NOT EXISTS maintenance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operation_type TEXT,  -- 'cleanup', 'backup', 'optimization'
    details TEXT,
    status TEXT,
    duration INTEGER,     -- in seconds
    bytes_affected INTEGER
);

-- Triggers to update timestamps
CREATE TRIGGER IF NOT EXISTS update_config_sections_timestamp 
AFTER UPDATE ON config_sections
BEGIN
    UPDATE config_sections SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_config_values_timestamp 
AFTER UPDATE ON config_values
BEGIN
    UPDATE config_values SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_cameras_timestamp 
AFTER UPDATE ON cameras
BEGIN
    UPDATE cameras SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_performance_metrics_camera_time 
ON performance_metrics(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_recognition_metrics_camera_time 
ON recognition_metrics(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_camera_time 
ON events(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_face_detections_camera_time 
ON face_detections(camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_error_logs_camera_time 
ON error_logs(camera_id, timestamp);

-- Views for common queries
CREATE VIEW IF NOT EXISTS v_camera_performance AS
SELECT 
    c.camera_id,
    c.name,
    AVG(pm.fps) as avg_fps,
    AVG(pm.processing_time) as avg_processing_time,
    AVG(pm.cpu_usage) as avg_cpu_usage,
    COUNT(DISTINCT fd.person_id) as unique_faces_detected,
    COUNT(e.id) as error_count
FROM cameras c
LEFT JOIN performance_metrics pm ON c.camera_id = pm.camera_id
LEFT JOIN face_detections fd ON c.camera_id = fd.camera_id
LEFT JOIN error_logs e ON c.camera_id = e.camera_id
GROUP BY c.camera_id, c.name; 