"""
Face matching module for ScaleTent
Handles face embedding generation and matching against a database of known faces
"""

import numpy as np
import torch
from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from src.core.logger import setup_logger
from src.core.device import get_device_from_config

logger = setup_logger(__name__)

class FaceMatcher:
    """Face matching implementation using FaceNet embeddings"""
    
    def __init__(
        self,
        embedder_model_path: str,
        database_path: str,
        similarity_threshold: float = 0.7,
        device: str = 'auto'
    ):
        """
        Initialize face matcher
        
        Args:
            embedder_model_path (str): Path to FaceNet model weights
            database_path (str): Path to face embeddings database
            similarity_threshold (float): Threshold for face similarity (0-1)
            device (str): Device to run inference on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.database_path = Path(database_path)
        self.similarity_threshold = similarity_threshold
        
        # Determine device
        if device == 'auto':
            self.device = get_device_from_config('auto')
        else:
            self.device = torch.device(device)
        
        # Initialize embedder
        self._load_embedder(embedder_model_path)
        
        # Load face database
        self.known_embeddings: Dict[str, Dict] = {}
        self._load_database()
        
        # Performance tracking
        self.embedding_times = []
        self.matching_times = []
        self.max_times = 100
        
        logger.info(f"Face matcher initialized with database: {database_path}")
        logger.info(f"Running on device: {self.device}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Known faces in database: {len(self.known_embeddings)}")
    
    def _load_embedder(self, model_path: str):
        """Load the FaceNet model"""
        try:
            logger.info("Loading FaceNet model")
            
            # First try to load model on specified device
            try:
                self.embedder = InceptionResnetV1(
                    pretrained='vggface2',
                    device=self.device
                ).eval()
            except (RuntimeError, AssertionError) as e:
                # If loading on specified device fails, fall back to CPU
                logger.warning(f"Failed to load model on {self.device}, falling back to CPU: {str(e)}")
                self.device = torch.device('cpu')
                self.embedder = InceptionResnetV1(
                    pretrained='vggface2',
                    device=self.device
                ).eval()
            
            logger.info("FaceNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {e}")
            raise
    
    def _load_database(self):
        """Load face embeddings database"""
        try:
            if self.database_path.exists():
                logger.info(f"Loading face database from {self.database_path}")
                with open(self.database_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_embeddings)} face embeddings")
            else:
                logger.info("No existing face database found. Starting with empty database.")
                self.known_embeddings = {}
        
        except Exception as e:
            logger.error(f"Failed to load face database: {e}")
            self.known_embeddings = {}
    
    def _save_database(self):
        """Save face embeddings database"""
        try:
            logger.info(f"Saving face database to {self.database_path}")
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            logger.info("Face database saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
    
    def generate_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for a face image
        
        Args:
            face_img (numpy.ndarray): Aligned face image
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        start_time = time.time()
        
        try:
            # Convert to PIL Image
            face_pil = Image.fromarray(face_img)
            
            # Preprocess
            face_tensor = torch.from_numpy(np.array(face_pil)).float()
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.embedder(face_tensor)
            
            # Convert to numpy
            embedding = embedding.cpu().numpy().flatten()
            
            # Record time
            embed_time = time.time() - start_time
            self.embedding_times.append(embed_time)
            if len(self.embedding_times) > self.max_times:
                self.embedding_times = self.embedding_times[-self.max_times:]
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return None
    
    def match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match a face embedding against the database
        
        Args:
            embedding (numpy.ndarray): Face embedding to match
        
        Returns:
            tuple: (person_id, similarity_score) or (None, 0.0) if no match
        """
        start_time = time.time()
        
        try:
            best_match = None
            best_score = 0.0
            
            for person_id, data in self.known_embeddings.items():
                known_embedding = data['embedding']
                
                # Calculate cosine similarity
                similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person_id
            
            # Record time
            match_time = time.time() - start_time
            self.matching_times.append(match_time)
            if len(self.matching_times) > self.max_times:
                self.matching_times = self.matching_times[-self.max_times:]
            
            # Return match if above threshold
            if best_score >= self.similarity_threshold:
                return best_match, best_score
            return None, 0.0
        
        except Exception as e:
            logger.error(f"Error matching face: {e}")
            return None, 0.0
    
    def add_face(self, person_id: str, face_img: np.ndarray, metadata: Dict = None) -> bool:
        """
        Add a new face to the database
        
        Args:
            person_id (str): Unique identifier for the person
            face_img (numpy.ndarray): Face image
            metadata (dict, optional): Additional metadata about the person
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(face_img)
            if embedding is None:
                return False
            
            # Add to database
            self.known_embeddings[person_id] = {
                'embedding': embedding,
                'added_at': time.time(),
                'metadata': metadata or {}
            }
            
            # Save database
            self._save_database()
            
            logger.info(f"Added new face to database: {person_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding face to database: {e}")
            return False
    
    def remove_face(self, person_id: str) -> bool:
        """
        Remove a face from the database
        
        Args:
            person_id (str): Person ID to remove
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if person_id in self.known_embeddings:
                del self.known_embeddings[person_id]
                self._save_database()
                logger.info(f"Removed face from database: {person_id}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Error removing face from database: {e}")
            return False
    
    def get_avg_embedding_time(self) -> float:
        """Get average embedding generation time in milliseconds"""
        if not self.embedding_times:
            return 0
        return sum(self.embedding_times) / len(self.embedding_times) * 1000
    
    def get_avg_matching_time(self) -> float:
        """Get average face matching time in milliseconds"""
        if not self.matching_times:
            return 0
        return sum(self.matching_times) / len(self.matching_times) * 1000

    def match(self, face_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Match a face image against the database
        
        Args:
            face_image (numpy.ndarray): Face image to match
            
        Returns:
            dict or None: Match result containing:
                - 'id': str - Identity ID
                - 'similarity': float - Similarity score
                - 'metadata': dict - Person metadata
                or None if no match found
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(face_image)
            if embedding is None:
                return None
            
            # Match against database
            identity_id, similarity = self.match_face(embedding)
            
            if identity_id is not None:
                person_data = self.known_embeddings[identity_id].copy()
                person_data.pop('embedding', None)  # Remove embedding from response
                
                return {
                    'id': identity_id,
                    'similarity': similarity,
                    'metadata': person_data.get('metadata', {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in match: {e}")
            return None

    def match_face_in_frame(self, frame: np.ndarray, face_detection: Dict[str, Any]) -> Tuple[bool, Optional[str], float, Optional[Dict]]:
        """
        Match a face detected in a frame against the database
        
        Args:
            frame (numpy.ndarray): Full frame containing the face
            face_detection (dict): Face detection data including bounding box
                Expected keys:
                - 'bbox': List[float] - [x1, y1, x2, y2] normalized coordinates
                - 'confidence': float - Detection confidence score
        
        Returns:
            tuple:
                - bool: Whether face was recognized
                - str or None: Identity ID if recognized, None otherwise  
                - float: Similarity score (0-1)
                - dict or None: Person metadata if recognized, None otherwise
        """
        try:
            # Extract face region
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = face_detection['bbox']
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            
            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                logger.warning("Invalid face bounding box")
                return False, None, 0.0, None
                
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                logger.warning("Empty face region extracted")
                return False, None, 0.0, None
            
            # Generate embedding
            embedding = self.generate_embedding(face_img)
            if embedding is None:
                return False, None, 0.0, None
            
            # Match against database
            identity_id, similarity = self.match_face(embedding)
            
            if identity_id is not None:
                # Get person metadata
                person_data = self.known_embeddings[identity_id].copy()
                # Remove embedding from metadata
                person_data.pop('embedding', None)
                return True, identity_id, similarity, person_data
            
            return False, None, similarity, None
            
        except Exception as e:
            logger.error(f"Error in match_face_in_frame: {e}")
            return False, None, 0.0, None