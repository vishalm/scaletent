#!/usr/bin/env python3
"""
ScaleTent - Download Models Script
Downloads all required models for the ScaleTent system
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
import hashlib
from pathlib import Path
import gdown
import requests
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize argument parser
parser = argparse.ArgumentParser(description='Download models for ScaleTent')
parser.add_argument('--models', nargs='+', choices=['yolo', 'face_detector', 'face_embedder', 'all'], 
                    default=['all'], help='Models to download')
parser.add_argument('--output', type=str, default='data/models', 
                    help='Output directory for downloaded models')
parser.add_argument('--force', action='store_true', 
                    help='Force download even if models already exist')
parser.add_argument('--cuda', action='store_true', 
                    help='Download CUDA-optimized models if available')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Model definitions
MODELS = {
    'yolo': {
        'urls': {
            'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
        },
        'md5': {
            'yolov8n': 'c55ae8c7c14a247d0112eabcc8952566',
            'yolov8s': '833dc69720a908e2e65642bb0e6a3645',
            'yolov8m': 'b2c1c1d71d7d12333425701cf622e672',
            'yolov8l': '78eb932173179ea9ef8ab3ffbba2d853',
            'yolov8x': 'd2f93a4a52fd917fb2cc6a353f360886',
        },
        'default': 'yolov8n'
    },
    'face_detector': {
        'urls': {
            'opencv_face_detector': {
                'model': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
                'config': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
            },
            'mediapipe_face_detector': None,  # Built-in to mediapipe
            'retinaface': 'https://drive.google.com/uc?id=1MJ5tK4GyVbMo67D-hGQ3jpQ-5Q3aWM_N'
        },
        'md5': {
            'opencv_face_detector': {
                'model': '4a2ffc9e4a7af8af5a09221f7ae9aff0',
                'config': '6d2d7e0f209c08af32e87bfadf9f2aaa'
            },
            'mediapipe_face_detector': None,
            'retinaface': 'e8b63c5e73cf866aea5ca1e4f2da6204'
        },
        'default': 'opencv_face_detector'
    },
    'face_embedder': {
        'urls': {
            'facenet': 'https://drive.google.com/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1',
            'arcface': 'https://drive.google.com/uc?id=1omDumNbVGwPkDlMn-1HgJJGXFIgc_0EQ',
            'insightface': 'https://drive.google.com/uc?id=1cUP6mVODTOWsKFKy3jD4BipjQRGOzMk2'
        },
        'md5': {
            'facenet': '110fd1078e10bc98a5e823cdcd7e4e8c',
            'arcface': 'e1907db7618c4da8c1e8c2335cb99159',
            'insightface': '50ac8b9df7e9de3b0f868a0883ff3262'
        },
        'default': 'facenet'
    }
}

def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)
                
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def download_from_gdrive(file_id, output_path, desc=None):
    """Download a file from Google Drive"""
    try:
        with tqdm(desc=desc, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output_path,
                quiet=True,
                fuzzy=True,
                progress=lambda x, y: progress_bar.update(y)
            )
        return True
    except Exception as e:
        print(f"Error downloading file from Google Drive: {e}")
        return False

def verify_md5(file_path, expected_md5):
    """Verify file integrity using MD5 hash"""
    if not expected_md5:
        return True
    
    print(f"Verifying {file_path}...")
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    
    if file_hash.hexdigest() == expected_md5:
        print(f"✅ Verification successful")
        return True
    else:
        print(f"❌ Verification failed. Expected: {expected_md5}, Got: {file_hash.hexdigest()}")
        return False

def download_yolo_models():
    """Download YOLOv8 models"""
    print("\n📥 Downloading YOLOv8 models...")
    
    model_size = args.model_size if hasattr(args, 'model_size') else MODELS['yolo']['default']
    model_url = MODELS['yolo']['urls'][model_size]
    model_md5 = MODELS['yolo']['md5'][model_size]
    
    output_path = os.path.join(args.output, f"{model_size}.pt")
    
    if os.path.exists(output_path) and not args.force:
        print(f"✅ YOLOv8 model already exists at {output_path}")
        if verify_md5(output_path, model_md5):
            return True
        else:
            print("❌ MD5 verification failed. Re-downloading...")
    
    success = download_file(model_url, output_path, f"Downloading {model_size}")
    
    if success:
        verify_md5(output_path, model_md5)
        print(f"✅ YOLOv8 model downloaded to {output_path}")
    else:
        print(f"❌ Failed to download YOLOv8 model")
    
    return success

def download_face_detector_models():
    """Download face detector models"""
    print("\n📥 Downloading face detector models...")
    
    detector_type = args.detector_type if hasattr(args, 'detector_type') else MODELS['face_detector']['default']
    
    if detector_type == 'mediapipe_face_detector':
        print("ℹ️ MediaPipe face detector is included with the mediapipe package.")
        print("✅ No download needed.")
        return True
    
    if detector_type == 'opencv_face_detector':
        model_info = MODELS['face_detector']['urls'][detector_type]
        model_md5 = MODELS['face_detector']['md5'][detector_type]
        
        # Download model file
        model_url = model_info['model']
        model_output = os.path.join(args.output, 'opencv_face_detector.caffemodel')
        
        if os.path.exists(model_output) and not args.force:
            print(f"✅ OpenCV face detector model already exists at {model_output}")
            if not verify_md5(model_output, model_md5['model']):
                print("❌ MD5 verification failed. Re-downloading...")
                if not download_file(model_url, model_output, "Downloading OpenCV face detector model"):
                    return False
        else:
            if not download_file(model_url, model_output, "Downloading OpenCV face detector model"):
                return False
        
        # Download config file
        config_url = model_info['config']
        config_output = os.path.join(args.output, 'deploy.prototxt')
        
        if os.path.exists(config_output) and not args.force:
            print(f"✅ OpenCV face detector config already exists at {config_output}")
            if not verify_md5(config_output, model_md5['config']):
                print("❌ MD5 verification failed. Re-downloading...")
                if not download_file(config_url, config_output, "Downloading OpenCV face detector config"):
                    return False
        else:
            if not download_file(config_url, config_output, "Downloading OpenCV face detector config"):
                return False
        
        return True
    
    if detector_type == 'retinaface':
        model_url = MODELS['face_detector']['urls'][detector_type]
        model_md5 = MODELS['face_detector']['md5'][detector_type]
        
        # Extract Google Drive ID from URL
        drive_id = model_url.split('id=')[1]
        output_path = os.path.join(args.output, 'retinaface_model.zip')
        extract_dir = os.path.join(args.output, 'retinaface')
        
        if os.path.exists(extract_dir) and not args.force:
            print(f"✅ RetinaFace model already exists at {extract_dir}")
            return True
        
        # Download and extract
        if not download_from_gdrive(drive_id, output_path, "Downloading RetinaFace model"):
            return False
        
        print(f"📦 Extracting RetinaFace model...")
        try:
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✅ RetinaFace model extracted to {extract_dir}")
            
            # Clean up zip file
            os.remove(output_path)
            return True
        except Exception as e:
            print(f"❌ Error extracting RetinaFace model: {e}")
            return False
    
    print(f"❌ Unknown face detector type: {detector_type}")
    return False

def download_face_embedder_models():
    """Download face embedder models"""
    print("\n📥 Downloading face embedder models...")
    
    embedder_type = args.embedder_type if hasattr(args, 'embedder_type') else MODELS['face_embedder']['default']
    model_url = MODELS['face_embedder']['urls'][embedder_type]
    model_md5 = MODELS['face_embedder']['md5'][embedder_type]
    
    if embedder_type == 'facenet':
        output_dir = os.path.join(args.output, 'facenet')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'facenet_model.pt')
        
        if os.path.exists(output_path) and not args.force:
            print(f"✅ FaceNet model already exists at {output_path}")
            if verify_md5(output_path, model_md5):
                return True
            else:
                print("❌ MD5 verification failed. Re-downloading...")
        
        # Extract Google Drive ID from URL
        drive_id = model_url.split('id=')[1]
        if not download_from_gdrive(drive_id, output_path, "Downloading FaceNet model"):
            return False
        
        verify_md5(output_path, model_md5)
        print(f"✅ FaceNet model downloaded to {output_path}")
        return True
    
    elif embedder_type == 'arcface':
        output_dir = os.path.join(args.output, 'arcface')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'arcface_model.zip')
        extract_dir = output_dir
        
        if os.path.isdir(extract_dir) and os.listdir(extract_dir) and not args.force:
            print(f"✅ ArcFace model already exists at {extract_dir}")
            return True
        
        # Extract Google Drive ID from URL
        drive_id = model_url.split('id=')[1]
        if not download_from_gdrive(drive_id, output_path, "Downloading ArcFace model"):
            return False
        
        print(f"📦 Extracting ArcFace model...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✅ ArcFace model extracted to {extract_dir}")
            
            # Clean up zip file
            os.remove(output_path)
            return True
        except Exception as e:
            print(f"❌ Error extracting ArcFace model: {e}")
            return False
    
    elif embedder_type == 'insightface':
        output_dir = os.path.join(args.output, 'insightface')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'insightface_model.zip')
        extract_dir = output_dir
        
        if os.path.isdir(extract_dir) and os.listdir(extract_dir) and not args.force:
            print(f"✅ InsightFace model already exists at {extract_dir}")
            return True
        
        # Extract Google Drive ID from URL
        drive_id = model_url.split('id=')[1]
        if not download_from_gdrive(drive_id, output_path, "Downloading InsightFace model"):
            return False
        
        print(f"📦 Extracting InsightFace model...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✅ InsightFace model extracted to {extract_dir}")
            
            # Clean up zip file
            os.remove(output_path)
            return True
        except Exception as e:
            print(f"❌ Error extracting InsightFace model: {e}")
            return False
    
    print(f"❌ Unknown face embedder type: {embedder_type}")
    return False

def main():
    """Main function"""
    print("🚀 ScaleTent Model Downloader")
    print(f"📁 Output directory: {args.output}")
    
    # Force mode warning
    if args.force:
        print("⚠️ Force mode enabled. Existing models will be overwritten.")
    
    # Download selected models
    success = True
    
    if 'all' in args.models or 'yolo' in args.models:
        success = download_yolo_models() and success
    
    if 'all' in args.models or 'face_detector' in args.models:
        success = download_face_detector_models() and success
    
    if 'all' in args.models or 'face_embedder' in args.models:
        success = download_face_embedder_models() and success
    
    # Final message
    if success:
        print("\n✅ All models downloaded successfully!")
        print(f"📁 Models are located in: {os.path.abspath(args.output)}")
    else:
        print("\n❌ Some models failed to download. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()