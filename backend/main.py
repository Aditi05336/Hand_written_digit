"""
FastAPI Application for Handwritten Digit Recognition
Includes comprehensive CORS, endpoint schemas, and robust error handling.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.model import (
    ModelManager,
    decode_image,
    preprocess_image,
    predict,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.main")

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Initialize FastAPI application
app = FastAPI(
    title="Handwritten Digit Recognition AI",
    description="Production-ready FastAPI backend serving a PyTorch MLP trained on MNIST.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
# Adheres strictly to W3C CORS Specification:
# When allow_origins=["*"] (wildcard), allow_credentials MUST be False.
# Configurable via ALLOWED_ORIGINS environment variable for custom domains.
raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
if raw_origins == "*":
    allowed_origins = ["*"]
    allow_credentials = False
else:
    allowed_origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", "X-Requested-With"],
    max_age=600,
)

logger.info(
    f"CORS configured: origins={allowed_origins}, credentials={allow_credentials}"
)

# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """
    Distinguishes between syntax-level JSON decode errors (HTTP 400 Bad Request)
    and semantic schema validation errors (HTTP 422 Unprocessable Entity).
    """
    for err in exc.errors():
        err_type = str(err.get("type", "")).lower()
        if "json" in err_type or "decode" in err_type:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Malformed JSON in request body."},
            )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
try:
    model_manager = ModelManager()
    logger.info("PyTorch digit recognition model successfully initialized.")
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    model_manager = None


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str = Field(
        ...,
        min_length=1,
        description="Base64 encoded image string or data URL (e.g. data:image/png;base64,...)",
    )


class PredictResponse(BaseModel):
    success: bool
    predicted_digit: int
    confidence: float
    probabilities: List[float]
    preprocessed_image: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: Optional[float] = None
    device: str


# Mount static assets for frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", summary="Serve Web Application Frontend", response_class=FileResponse)
def read_root():
    """Serves the frontend single-page interface index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/health", response_model=HealthResponse, summary="Application Health Check")
def health_check():
    """Returns application health and model readiness status."""
    is_loaded = model_manager is not None and model_manager.is_ready()
    accuracy = model_manager.metadata.get("accuracy") if is_loaded else None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_accuracy=accuracy,
        device="cpu",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Handwritten Digit",
)
def predict_digit_endpoint(request: PredictRequest):
    """
    Receives canvas drawing, applies MNIST preprocessing, and predicts the digit.
    Includes full preprocessing debug stages and pixel normalization statistics.
    """
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load checkpoint.",
        )

    try:
        # Decode base64 image
        pil_image = decode_image(request.image)

        # Apply robust MNIST preprocessing with full debug pipeline
        tensor, preview_b64, debug_info = preprocess_image(pil_image)

        # Run model inference
        result = predict(model_manager.model, tensor)

        return PredictResponse(
            success=True,
            predicted_digit=result["predicted_digit"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            preprocessed_image=preview_b64,
            debug=debug_info,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as e:
        logger.exception("Inference error occurred during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Quietly handle favicon requests without 404 noise."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Fallback routing for style.css and script.js when accessed at root path
@app.get("/{filename}.css", include_in_schema=False)
def get_css(filename: str):
    file_path = FRONTEND_DIR / f"{filename}.css"
    if file_path.exists():
        return FileResponse(file_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/{filename}.js", include_in_schema=False)
def get_js(filename: str):
    file_path = FRONTEND_DIR / f"{filename}.js"
    if file_path.exists():
        return FileResponse(file_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")


@app.options("/", include_in_schema=False)
@app.options("/health", include_in_schema=False)
@app.options("/predict", include_in_schema=False)
def route_options():
    """Handles generic HTTP OPTIONS requests that don't include CORS headers."""
    return Response(status_code=status.HTTP_200_OK)
