"""
Pydantic models for the embedding API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    input_ids: List[List[int]] = Field(
        ...,
        description="Tokenized input IDs from the tokenizer",
        example=[[101, 7592, 2088, 102]]
    )
    attention_mask: List[List[int]] = Field(
        ...,
        description="Attention mask for the input",
        example=[[1, 1, 1, 1]]
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embedding: List[List[float]] = Field(
        ...,
        description="Generated embeddings",
        example=[[0.1, -0.2, 0.3]]
    )
    shape: List[int] = Field(
        ...,
        description="Shape of the embedding tensor",
        example=[1, 384]
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    model_name: str = Field(..., description="Loaded model name")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")