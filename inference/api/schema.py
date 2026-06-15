from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool = False
    load_error: str | None = None


class DetectRequest(BaseModel):
    """Request model for URL-based detection."""

    url: str


class DetectResponse(BaseModel):
    """Response model for detection."""

    image_id: str
    is_screen: bool


class ClassifyRequest(BaseModel):
    """Request model for updating image classification."""

    image_id: str
    is_screen: bool


class ClassifyResponse(BaseModel):
    """Response model for class update."""

    image_id: str
    is_screen: bool
    class_name: str


class PackageRequest(BaseModel):
    """Request model for packaging images after a timestamp."""

    after_timestamp: datetime
