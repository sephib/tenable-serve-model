"""
Embedding service for handling model loading and inference.
"""

import torch
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from config import settings


class EmbeddingService:
    """Service for managing the embedding model and performing inference."""

    def __init__(self):
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_name: str = ""
        self.device: str = "cpu"
        self._is_loaded: bool = False

    def _detect_device(self) -> str:
        """
        Automatically detect the best available device.

        Returns:
            Device string: "cuda" if GPU available, "cpu" otherwise
        """
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"🎮 GPU detected: {gpu_name} (Count: {gpu_count})")
            return device
        else:
            print("🖥️ No GPU detected, using CPU")
            return "cpu"

    def load_model(self) -> None:
        """Load the model and tokenizer from cache or download."""
        try:
            # Get configuration
            model_config = settings.get('MODEL', settings.get('model', {}))

            self.model_name = model_config.get('name', 'intfloat/multilingual-e5-small')

            # Automatically detect the best available device
            self.device = self._detect_device()

            # Get data directory and model path
            data_dir = settings.get('data_dir', '../data')
            script_dir = Path(__file__).parent
            data_path = script_dir / data_dir

            local_path_setting = model_config.get('local_path', 'models/multilingual-e5-small')
            if local_path_setting.startswith('../data/'):
                cache_dir = data_path / local_path_setting.lstrip('../data/')
            else:
                cache_dir = script_dir / local_path_setting

            print(f"🔄 Loading model: {self.model_name}")
            print(f"📁 Cache directory: {cache_dir}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir)
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(cache_dir)
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            print(f"✅ Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        print("🗑️ Model unloaded")

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._is_loaded and self.model is not None and self.tokenizer is not None

    def get_embeddings(self, input_ids: List[List[int]], attention_mask: List[List[int]]) -> Tuple[List[List[float]], List[int]]:
        """
        Generate embeddings from tokenized input.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for the input

        Returns:
            Tuple of (embeddings, shape)
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        try:
            # Convert to tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor
                )

                # Use mean pooling over the sequence dimension
                embeddings = outputs.last_hidden_state

                # Apply attention mask and mean pooling
                mask_expanded = attention_mask_tensor.unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = embeddings * mask_expanded
                embeddings = embeddings.sum(1) / mask_expanded.sum(1)

            # Convert to list for JSON serialization
            embeddings_list = embeddings.cpu().numpy().tolist()
            shape = list(embeddings.shape)

            return embeddings_list, shape

        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")


# Global service instance
embedding_service = EmbeddingService()