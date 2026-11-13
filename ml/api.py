
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import load_model, predict_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="CyberGuard API", version="1.0.0", lifespan=lifespan)


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=10000)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Random Forest"}


@app.post("/predict")
def predict(request: URLRequest):
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only HTTP and HTTPS URLs are supported")

    try:
        return predict_url(url, app.state.model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
