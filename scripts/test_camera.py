#!/usr/bin/env python3
"""
ScaleTent - Camera Testing Script
Tests camera connections and displays video feed with YOLOv8 detection
"""

import os
import sys
import argparse
import cv2
import time
import yaml
import numpy as np
from pathlib import Path
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Parse arguments
parser = argparse.ArgumentParser(description='Test camera connections')
parser.add_argument('--source', type=str, default='0', help='Camera source (number, URL, file path)')
parser.add_argument('--width', type=int, default=1280, help='Camera width')
parser.add_argument('--height', type=int, default=720, help='Camera height')
parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
parser.add_argument('--detection', action='store_true', help='Enable YOLO detection')
parser.add_argument('--face', action='store_true', help='Enable face detection')
parser.add_argument('--output', type=str, default=None, help='Output video file path')
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
parser.add_argument('--save-config', action='store_true', help='Save camera configuration to config file')
parser.add_argument('--camera-id', type=str, default='camera-test', help='Camera ID for config')
parser.add_argument('--time', type=int, default=0, help='Run for specified time in seconds (0 = infinite)')
args = parser.parse_args()

# Global variables
running = True
frame_count = 0
start_time = 0
current_fps = 0
detector = None
face_detector = None

def load_yolo():
    """Load YOLOv8 model for object detection"""
    print("Loading YOLOv8 model...")
    try:
        from ultralytics import YOLO
        
        # Check if model exists in data/models
        model_path = Path("data/models/yolov8n.pt")
        if not model_path.exists():
            print(f"Model not found at {model_path}, downloading...")
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(model_path)
        
        print("YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return None

def load_face_detector():
    """Load face detection model"""
    print("Loading face detection model...")
    try:
        # Try to use OpenCV's face detector
        model_path = "data/models/opencv_face_detector.caffemodel"
        config_path = "data/models/deploy.prototxt"
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Face detection model not found. Downloading...")
            os.makedirs("data/models", exist_ok=True)
            
            # Download model file
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                model_path
            )
            
            # Download config file
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                config_path
            )
        
        # Load the model
        face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        
        print("Face detection model loaded successfully")
        return face_net
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        return None

def detect_objects(frame, model):
    """Detect objects in frame using YOLOv8"""
    try:
        # Run inference
        results = model(frame, verbose=False)
        
        # Get bounding boxes
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Class 0 is person in COCO dataset
                if cls == 0 and conf > 0.25:
                    boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': 'person'
                    })
        
        return boxes
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []

def detect_faces(frame, face_net):
    """Detect faces in frame using OpenCV DNN"""
    try:
        # Prepare image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Inference
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Get bounding boxes
        boxes = []
        height, width = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                
                boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence)
                })
        
        return boxes
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def draw_boxes(frame, boxes, color=(0, 255, 0), is_face=False):
    """Draw bounding boxes on frame"""
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        conf = box['confidence']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if is_face:
            label = f"Face: {conf:.2f}"
        else:
            label = f"{box.get('class_name', 'Object')}: {conf:.2f}"
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return frame

def process_frame(frame):
    """Process frame with detection models"""
    global detector, face_detector
    
    # Detection copies
    detection_frame = frame.copy()
    result_frame = frame.copy()
    
    # Object detection
    if args.detection and detector is not None:
        boxes = detect_objects(detection_frame, detector)
        result_frame = draw_boxes(result_frame, boxes, color=(0, 255, 0))
    
    # Face detection
    if args.face and face_detector is not None:
        faces = detect_faces(detection_frame, face_detector)
        result_frame = draw_boxes(result_frame, faces, color=(0, 0, 255), is_face=True)
    
    return result_frame

def update_fps():
    """Update FPS calculation"""
    global frame_count, start_time, current_fps
    
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    # Update FPS every second
    if elapsed_time >= 1.0:
        current_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

def save_camera_config():
    """Save camera configuration to config file"""
    if not args.save_config:
        return
    
    try:
        config_path = Path(args.config)
        
        # Load existing config if it exists
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Initialize cameras section if it doesn't exist
        if 'cameras' not in config:
            config['cameras'] = []
        
        # Check if camera with this ID already exists
        camera_exists = False
        for i, camera in enumerate(config['cameras']):
            if camera.get('id') == args.camera_id:
                # Update existing camera
                config['cameras'][i] = {
                    'id': args.camera_id,
                    'source': args.source,
                    'width': args.width,
                    'height': args.height,
                    'fps': args.fps,
                    'enabled': True
                }
                camera_exists = True
                break
        
        # Add new camera if it doesn't exist
        if not camera_exists:
            config['cameras'].append({
                'id': args.camera_id,
                'source': args.source,
                'width': args.width,
                'height': args.height,
                'fps': args.fps,
                'enabled': True
            })
        
        # Save config
        os.makedirs(config_path.parent, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Camera configuration saved to {config_path}")
    
    except Exception as e:
        print(f"Error saving camera configuration: {e}")

def main():
    """Main function"""
    global running, start_time, detector, face_detector
    
    print(f"Testing camera: {args.source}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    
    # Initialize video capture
    if args.source.isdigit():
        # Camera ID
        cap = cv2.VideoCapture(int(args.source))
    else:
        # URL or file path
        cap = cv2.VideoCapture(args.source)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.source}")
        return 1
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Actual resolution: {actual_width}x{actual_height}")
    print(f"Actual FPS: {actual_fps}")
    
    # Initialize video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            actual_fps,
            (actual_width, actual_height)
        )
        print(f"Recording video to {args.output}")
    
    # Load detection models if needed
    if args.detection:
        detector = load_yolo()
        if detector is None:
            print("Warning: Object detection disabled due to model loading error")
    
    if args.face:
        face_detector = load_face_detector()
        if face_detector is None:
            print("Warning: Face detection disabled due to model loading error")
    
    # Initialize FPS calculation
    start_time = time.time()
    run_start_time = time.time()
    
    # Main loop
    try:
        while running:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = process_frame(frame)
            
            # Update FPS calculation
            update_fps()
            
            # Add FPS display
            cv2.putText(
                processed_frame,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow("Camera Test", processed_frame)
            
            # Write frame if recording
            if writer is not None:
                writer.write(processed_frame)
            
            # Exit on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check time limit
            if args.time > 0 and (time.time() - run_start_time) > args.time:
                print(f"Time limit of {args.time} seconds reached")
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error in main loop: {e}")
    
    finally:
        # Clean up
        running = False
        cap.release()
        
        if writer is not None:
            writer.release()
        
        cv2.destroyAllWindows()
        
        # Save camera configuration if requested
        save_camera_config()
        
        print("Camera test completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())