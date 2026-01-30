import base64
from html import escape


from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from .utils import setup_logger
from .predictor import Predictor

app = FastAPI(title="CIFAR-10 CNN Classifier")

logger = setup_logger("cifar10_app", "logs/app.log")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "cnn_model.pth"

# Load model once when server starts
try:
    predictor = Predictor(model_path=str(MODEL_PATH), logger=logger)
except Exception:
    predictor = None
    logger.exception("--- Failed to initialize Predictor ---")

@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(
            status_code=500, 
            detail=f"---Model not loaded. Expected at: {MODEL_PATH}---"
            )
    return {"status": "ok", "model_path": str(MODEL_PATH)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>CIFAR-10 Classifier</title>
      </head>
      <body style="font-family: Arial; max-width: 760px; margin: 40px auto; line-height:1.5;">
        <h2>CIFAR-10 Image Classifier</h2>
        <p>Upload an image to classify into one of 10 CIFAR-10 classes.</p>

        <form action="/predict" enctype="multipart/form-data" method="post"
              style="padding:16px; border:1px solid #ddd; border-radius:12px;">
          <input name="file" type="file" accept="image/*" required />
          <button type="submit" style="margin-left:10px; padding:6px 12px;">Predict</button>
        </form>

                
      </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    # Read image bytes
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception:
        logger.exception("Failed to read image upload")
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

    # Run prediction
    try:
        result = predictor.predict(image, top_k=3)
        top_pred = result["top_prediction"]["label"]
        topk = result["top_k"]

        logger.info(f"Predicted={top_pred} file={file.filename}")

        # Convert to % for display
        topk_rows = ""
        for r in topk:
            label = escape(r["label"])
            pct = r["probability"] * 100
            topk_rows += f"""
              <tr>
                <td style="padding:8px; border-bottom:1px solid #eee;">{label}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">{pct:.2f}%</td>
              </tr>
            """

        # Show uploaded image preview using base64
        b64 = base64.b64encode(content).decode("utf-8")
        mime = escape(file.content_type)
        filename = escape(file.filename or "uploaded_image")

        return f"""
        <html>
          <head><title>Prediction Result</title></head>
          <body style="font-family: Arial; max-width: 760px; margin: 40px auto; line-height:1.5;">
            <h2>Prediction Result</h2>

            <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;">
              <div>
                <h4 style="margin-bottom:8px;">Uploaded Image</h4>
                <img src="data:{mime};base64,{b64}" alt="uploaded" 
                     style="max-width:320px; border:1px solid #ddd; border-radius:12px;" />
                <p style="color:#666; font-size:13px;">{filename}</p>
              </div>

              <div style="min-width:320px;">
                <h4 style="margin-bottom:8px;">Top Prediction</h4>
                <div style="padding:14px; border:1px solid #ddd; border-radius:12px;">
                  <div style="font-size:18px;"><b>{escape(top_pred)}</b></div>
                </div>

                <h4 style="margin-top:18px; margin-bottom:8px;">Top-3 Probabilities</h4>
                <table style="width:100%; border-collapse:collapse; border:1px solid #ddd; border-radius:12px; overflow:hidden;">
                  <thead>
                    <tr style="background:#f7f7f7;">
                      <th style="text-align:left; padding:10px; border-bottom:1px solid #ddd;">Class</th>
                      <th style="text-align:left; padding:10px; border-bottom:1px solid #ddd;">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topk_rows}
                  </tbody>
                </table>

                <div style="margin-top:18px;">
                  <a href="/" style="text-decoration:none;">
                    <button style="padding:8px 14px;">Try Another Image</button>
                  </a>
                </div>
              </div>
            </div>
          </body>
        </html>
        """

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to server error")
