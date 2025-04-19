"""
Face embedding module for ScaleTent
Handles extraction of face embeddings for recognition
"""

import os
import cv2
import numpy as np
import time
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from core.logger import setup_logger

logger = setup_logger(__name__)

class FaceEmbedder:
    """
    Face embedding generator for ScaleTent
    
    This class handles extraction of face embeddings using different models:
    1. FaceNet
    2. ArcFace
    3. InsightFace
    4. OpenCV DNN
    5. MobileFaceNet
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face embedder
        
        Args:
            config (dict): Embedder configuration
        """
        self.config = config
        
        # Model settings
        self.model_path = Path(config.get("model_path", "data/models/facenet_model.pt"))
        self.model_type = config.get("model_type", "facenet")
        self.device = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = config.get("image_size", (160, 160))  # Default for FaceNet
        
        # Image preprocessing settings
        self.mean = config.get("mean", [0.5, 0.5, 0.5])
        self.std = config.get("std", [0.5, 0.5, 0.5])
        
        # Load model
        self.model = None
        self.face_processor = None
        self._load_model()
        
        # Performance tracking
        self.inference_times = []
        self.max_inference_times = 100
        
        logger.info(f"Face embedder initialized with model type: {self.model_type}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load face embedding model"""
        try:
            if self.model_type == "facenet":
                self._load_facenet_model()
            elif self.model_type == "arcface":
                self._load_arcface_model()
            elif self.model_type == "insightface":
                self._load_insightface_model()
            elif self.model_type == "opencv_dnn":
                self._load_opencv_dnn_model()
            elif self.model_type == "mobilefacenet":
                self._load_mobilefacenet_model()
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Error loading face embedding model: {e}")
            raise
    
    def _load_facenet_model(self):
        """Load FaceNet model"""
        try:
            from facenet_pytorch import InceptionResnetV1
            
            # Check if model file exists
            if not os.path.exists(self.model_path) and str(self.model_path).endswith('.pt'):
                logger.info("Using pretrained FaceNet model")
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
            else:
                logger.info(f"Loading FaceNet model from {self.model_path}")
                self.model = InceptionResnetV1(pretrained=None).eval()
                self.model.load_state_dict(torch.load(self.model_path))
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set image size
            self.image_size = (160, 160)
            
            logger.info("FaceNet model loaded successfully")
            
        except ImportError:
            logger.error("Failed to import facenet_pytorch. Please install it with: pip install facenet-pytorch")
            raise
        except Exception as e:
            logger.error(f"Error loading FaceNet model: {e}")
            raise
    
    def _load_arcface_model(self):
        """Load ArcFace model"""
        try:
            import mxnet as mx
            
            # Check if we have model file with ".params" extension (MXNet model)
            model_files = list(Path(self.model_path).parent.glob('*.params'))
            if not model_files:
                logger.error(f"No ArcFace model found at {self.model_path.parent}")
                raise FileNotFoundError(f"No ArcFace model found at {self.model_path.parent}")
            
            ctx = mx.gpu() if self.device.startswith('cuda') and mx.context.num_gpus() > 0 else mx.cpu()
            
            # Load model
            self.model = mx.gluon.SymbolBlock.imports(
                str(model_files[0]).replace('.params', '-symbol.json'),
                ['data'],
                str(model_files[0])
            )
            self.model.collect_params().reset_ctx(ctx)
            
            # Set image size (ArcFace uses 112x112)
            self.image_size = (112, 112)
            
            # Set preprocessing values for ArcFace
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
            
            logger.info("ArcFace model loaded successfully")
            
        except ImportError:
            logger.error("Failed to import mxnet. Please install it with: pip install mxnet-cu102 or mxnet")
            raise
        except Exception as e:
            logger.error(f"Error loading ArcFace model: {e}")
            raise
    
    def _load_insightface_model(self):
        """Load InsightFace model"""
        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
            
            # Check if we have a specific model file
            if os.path.exists(self.model_path):
                # Load custom InsightFace model
                logger.info(f"Loading InsightFace model from {self.model_path}")
                provider = 'CUDAExecutionProvider' if self.device.startswith('cuda') else 'CPUExecutionProvider'
                self.model = get_model(str(self.model_path), providers=[provider])
                self.model.prepare(ctx_id=0 if self.device.startswith('cuda') else -1)
            else:
                # Use the default InsightFace model
                logger.info("Using default InsightFace model")
                self.face_processor = FaceAnalysis(
                    providers=['CUDAExecutionProvider' if self.device.startswith('cuda') else 'CPUExecutionProvider']
                )
                self.face_processor.prepare(ctx_id=0 if self.device.startswith('cuda') else -1)
            
            # Set image size (InsightFace typically uses 112x112)
            self.image_size = (112, 112)
            
            logger.info("InsightFace model loaded successfully")
            
        except ImportError:
            logger.error("Failed to import insightface. Please install it with: pip install insightface")
            raise
        except Exception as e:
            logger.error(f"Error loading InsightFace model: {e}")
            raise
    
    def _load_opencv_dnn_model(self):
        """Load OpenCV DNN face recognition model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"OpenCV DNN model not found at {self.model_path}")
                raise FileNotFoundError(f"OpenCV DNN model not found at {self.model_path}")
            
            logger.info(f"Loading OpenCV DNN model from {self.model_path}")
            self.model = cv2.dnn.readNetFromTorch(str(self.model_path))
            
            # Set device
            if self.device.startswith('cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("Using CUDA backend for OpenCV DNN")
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                logger.info("Using CPU backend for OpenCV DNN")
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Set image size (OpenCV DNN models usually expect 224x224)
            self.image_size = (224, 224)
            
            # Set preprocessing values for OpenCV DNN
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            
            logger.info("OpenCV DNN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading OpenCV DNN model: {e}")
            raise
    
    def _load_mobilefacenet_model(self):
        """Load MobileFaceNet model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"MobileFaceNet model not found at {self.model_path}")
                raise FileNotFoundError(f"MobileFaceNet model not found at {self.model_path}")
            
            # Load model with ONNX Runtime
            import onnxruntime as ort
            
            logger.info(f"Loading MobileFaceNet model from {self.model_path}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.startswith('cuda') else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(str(self.model_path), providers=providers)
            
            # Set image size (MobileFaceNet typically uses 112x112)
            self.image_size = (112, 112)
            
            # Set preprocessing values
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            
            logger.info("MobileFaceNet model loaded successfully")
            
        except ImportError:
            logger.error("Failed to import onnxruntime. Please install it with: pip install onnxruntime-gpu or onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Error loading MobileFaceNet model: {e}")
            raise
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            face_image (numpy.ndarray): Face image (BGR format)
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        try:
            if face_image is None or face_image.size == 0:
                logger.warning("Empty face image provided")
                return np.zeros(512)  # Return zero embedding
            
            # Start timing
            start_time = time.time()
            
            # Preprocess face image
            preprocessed = self._preprocess_face(face_image)
            
            # Get embedding based on model type
            if self.model_type == "facenet":
                embedding = self._get_facenet_embedding(preprocessed)
            elif self.model_type == "arcface":
                embedding = self._get_arcface_embedding(preprocessed)
            elif self.model_type == "insightface":
                embedding = self._get_insightface_embedding(face_image)  # Uses original image
            elif self.model_type == "opencv_dnn":
                embedding = self._get_opencv_dnn_embedding(preprocessed)
            elif self.model_type == "mobilefacenet":
                embedding = self._get_mobilefacenet_embedding(preprocessed)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return np.zeros(512)  # Return zero embedding
            
            # Normalize embedding (L2 norm)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last N inference times
            if len(self.inference_times) > self.max_inference_times:
                self.inference_times = self.inference_times[-self.max_inference_times:]
            
            logger.debug(f"Face embedding generated in {inference_time:.4f} seconds")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return np.zeros(512)  # Return zero embedding in case of error
    
    def get_batch_embeddings(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings from a batch of face images
        
        Args:
            face_images (list): List of face images (BGR format)
        
        Returns:
            list: List of face embedding vectors
        """
        try:
            if not face_images:
                return []
            
            # Start timing
            start_time = time.time()
            
            # Preprocess face images
            preprocessed_batch = [self._preprocess_face(face) for face in face_images]
            
            # Get embeddings based on model type
            if self.model_type == "facenet":
                embeddings = self._get_facenet_batch_embeddings(preprocessed_batch)
            elif self.model_type == "arcface":
                embeddings = self._get_arcface_batch_embeddings(preprocessed_batch)
            elif self.model_type == "insightface":
                embeddings = self._get_insightface_batch_embeddings(face_images)  # Uses original images
            elif self.model_type == "opencv_dnn":
                # OpenCV DNN doesn't support batch processing, so process each image individually
                embeddings = [self._get_opencv_dnn_embedding(img) for img in preprocessed_batch]
            elif self.model_type == "mobilefacenet":
                embeddings = self._get_mobilefacenet_batch_embeddings(preprocessed_batch)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return [np.zeros(512) for _ in face_images]  # Return zero embeddings
            
            # Normalize embeddings (L2 norm)
            embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last N inference times
            if len(self.inference_times) > self.max_inference_times:
                self.inference_times = self.inference_times[-self.max_inference_times:]
            
            logger.debug(f"Batch of {len(face_images)} face embeddings generated in {inference_time:.4f} seconds")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch face embeddings: {e}")
            return [np.zeros(512) for _ in face_images]  # Return zero embeddings in case of error
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for embedding extraction
        
        Args:
            face_image (numpy.ndarray): Face image (BGR format)
        
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        # Resize image to target size
        face_image = cv2.resize(face_image, self.image_size)
        
        # Convert BGR to RGB (required for most models)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        face_image = face_image.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        face_image = (face_image - mean) / std
        
        return face_image
    
    def _get_facenet_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        """
        Get embedding using FaceNet model
        
        Args:
            preprocessed_face (numpy.ndarray): Preprocessed face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        # Convert to PyTorch tensor
        # HWC to CHW format (PyTorch uses channels first)
        face_tensor = torch.from_numpy(preprocessed_face.transpose(2, 0, 1)).unsqueeze(0)
        
        # Move to device
        face_tensor = face_tensor.to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy()[0]
        
        return embedding
    
    def _get_facenet_batch_embeddings(self, preprocessed_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get embeddings for a batch using FaceNet model
        
        Args:
            preprocessed_faces (list): List of preprocessed face images
        
        Returns:
            list: List of face embedding vectors
        """
        # Convert to batch of PyTorch tensors
        batch_tensors = []
        for face in preprocessed_faces:
            # HWC to CHW format (PyTorch uses channels first)
            face_tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0)
            batch_tensors.append(face_tensor)
        
        # Concatenate batch
        batch = torch.cat(batch_tensors, dim=0)
        
        # Move to device
        batch = batch.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        
        return embeddings
    
    def _get_arcface_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        """
        Get embedding using ArcFace model
        
        Args:
            preprocessed_face (numpy.ndarray): Preprocessed face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        import mxnet as mx
        
        # Convert to MXNet NDArray
        # HWC to CHW format (MXNet uses channels first)
        face_tensor = mx.nd.array(preprocessed_face.transpose(2, 0, 1)).expand_dims(0)
        
        # Get embedding
        embedding = self.model(face_tensor).asnumpy()[0]
        
        return embedding
    
    def _get_arcface_batch_embeddings(self, preprocessed_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get embeddings for a batch using ArcFace model
        
        Args:
            preprocessed_faces (list): List of preprocessed face images
        
        Returns:
            list: List of face embedding vectors
        """
        import mxnet as mx
        
        # Convert to batch of MXNet NDArrays
        batch_tensors = []
        for face in preprocessed_faces:
            # HWC to CHW format (MXNet uses channels first)
            face_tensor = mx.nd.array(face.transpose(2, 0, 1)).expand_dims(0)
            batch_tensors.append(face_tensor)
        
        # Concatenate batch
        batch = mx.nd.concat(*batch_tensors, dim=0)
        
        # Get embeddings
        embeddings = self.model(batch).asnumpy()
        
        return embeddings
    
    def _get_insightface_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get embedding using InsightFace model
        
        Args:
            face_image (numpy.ndarray): Original face image (BGR format)
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        if self.face_processor is not None:
            # Use FaceAnalysis app
            faces = self.face_processor.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected by InsightFace")
                return np.zeros(512)
            
            # Use the first face's embedding
            embedding = faces[0].embedding
            
        else:
            # Use custom model
            # Resize and preprocess according to model requirements
            preprocessed = cv2.resize(face_image, self.image_size)
            
            # Get embedding
            embedding = self.model.get_embedding(preprocessed)
        
        return embedding
    
    def _get_insightface_batch_embeddings(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get embeddings for a batch using InsightFace model
        
        Args:
            face_images (list): List of original face images (BGR format)
        
        Returns:
            list: List of face embedding vectors
        """
        embeddings = []
        
        for face in face_images:
            # InsightFace doesn't support true batching in the same way as PyTorch/TensorFlow
            # Process each image individually
            embedding = self._get_insightface_embedding(face)
            embeddings.append(embedding)
        
        return embeddings
    
    def _get_opencv_dnn_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        """
        Get embedding using OpenCV DNN model
        
        Args:
            preprocessed_face (numpy.ndarray): Preprocessed face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        # Prepare blob from image
        # OpenCV DNN expects channels-first format
        blob = cv2.dnn.blobFromImage(
            preprocessed_face,
            1.0,  # scalefactor
            self.image_size,  # size
            (0, 0, 0),  # mean (already normalized)
            swapRB=False,  # already RGB
            crop=False
        )
        
        # Set input to the model
        self.model.setInput(blob)
        
        # Get embedding
        embedding = self.model.forward()
        
        # Flatten embedding
        embedding = embedding.flatten()
        
        return embedding
    
    def _get_mobilefacenet_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        """
        Get embedding using MobileFaceNet ONNX model
        
        Args:
            preprocessed_face (numpy.ndarray): Preprocessed face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        # Prepare input for ONNX Runtime
        # HWC to CHW format (ONNX models usually expect channels first)
        input_data = preprocessed_face.transpose(2, 0, 1).astype(np.float32)
        
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        # Get model input name
        input_name = self.model.get_inputs()[0].name
        
        # Run inference
        outputs = self.model.run(None, {input_name: input_data})
        
        # Get embedding from output
        embedding = outputs[0][0]
        
        return embedding
    
    def _get_mobilefacenet_batch_embeddings(self, preprocessed_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get embeddings for a batch using MobileFaceNet ONNX model
        
        Args:
            preprocessed_faces (list): List of preprocessed face images
        
        Returns:
            list: List of face embedding vectors
        """
        # Prepare batch input for ONNX Runtime
        batch_input = np.zeros((len(preprocessed_faces), 3, self.image_size[0], self.image_size[1]), dtype=np.float32)
        
        for i, face in enumerate(preprocessed_faces):
            # HWC to CHW format (ONNX models usually expect channels first)
            batch_input[i] = face.transpose(2, 0, 1).astype(np.float32)
        
        # Get model input name
        input_name = self.model.get_inputs()[0].name
        
        # Run inference
        outputs = self.model.run(None, {input_name: batch_input})
        
        # Get embeddings from output
        embeddings = outputs[0]
        
        return embeddings
    
    def get_avg_inference_time(self) -> float:
        """
        Get average inference time in milliseconds
        
        Returns:
            float: Average inference time in ms
        """
        if not self.inference_times:
            return 0.0
        
        return sum(self.inference_times) / len(self.inference_times) * 1000
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """
        Compare two face images and return similarity score
        
        Args:
            face1 (numpy.ndarray): First face image
            face2 (numpy.ndarray): Second face image
        
        Returns:
            float: Similarity score (0.0-1.0, higher is more similar)
        """
        # Get embeddings
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def compare_embedding_to_face(self, embedding: np.ndarray, face: np.ndarray) -> float:
        """
        Compare an embedding to a face image
        
        Args:
            embedding (numpy.ndarray): Face embedding
            face (numpy.ndarray): Face image
        
        Returns:
            float: Similarity score (0.0-1.0, higher is more similar)
        """
        # Get embedding for the face
        face_emb = self.get_embedding(face)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding, face_emb) / (np.linalg.norm(embedding) * np.linalg.norm(face_emb))
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two face embeddings
        
        Args:
            emb1 (numpy.ndarray): First embedding
            emb2 (numpy.ndarray): Second embedding
        
        Returns:
            float: Similarity score (0.0-1.0, higher is more similar)
        """
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity