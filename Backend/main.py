"""FastAPI application that serves ReviewSense AI predictions."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.inference import ReviewSensePredictor


# Load the serialized model once when the API process starts, not for each request.
predictor = ReviewSensePredictor()

app = FastAPI(
    title="ReviewSense AI",
    description="Product review sentiment prediction API.",
    version="0.1.0",
)

# Allow the local frontend server to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ReviewRequest(BaseModel):
    """Text fields accepted by the prediction endpoint."""

    review: str = Field(default="", description="Full product review text")
    summary: str = Field(default="", description="Short review headline or summary")


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    """Simple check that the API started and loaded its model."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest) -> dict:
    """Return sentiment and the model's estimated class probabilities."""
    try:
        return predictor.predict(review=request.review, summary=request.summary)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
