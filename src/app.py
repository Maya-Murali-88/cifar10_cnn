from pathlib import Path
import base64
from html import escape
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

from .utils import setup_logger
from .predictor import Predictor

app = FastAPI(title="CIFAR-10 CNN Classifier")
logger = setup_logger("cifar10_app", "logs/app.log")

# Always resolve model path from project root (reliable with uvicorn reload)
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "cnn_model.pth"

# Load model once when server starts
try:
    predictor = Predictor(model_path=str(MODEL_PATH), logger=logger)
except Exception:
    predictor = None
    logger.exception(" Failed to initialize Predictor")


def render_page(result_html: str = "", error_msg: str = "") -> str:
    """Render the themed home page. Optionally show prediction result or an error box."""
    error_box = ""
    if error_msg:
        error_box = f"""
        <div style="margin-top:16px; padding:14px; border-radius:14px; border:1px solid #fecaca; background:#fff1f2; color:#991b1b;">
          <b>Error:</b> {escape(error_msg)}
        </div>
        """

    model_status = ""
    if predictor is None:
        model_status = f"""
        <div style="margin-top:14px; padding:12px; border-radius:14px; border:1px solid #fde68a; background:#fffbeb; color:#92400e;">
          <b>Model not loaded.</b> Expected at: <code>{escape(str(MODEL_PATH))}</code><br/>
          Train/save your model to <code>artifacts/cnn_model.pth</code> and restart the server.
        </div>
        """
    else:
        model_status = f"""
        <div style="margin-top:14px; padding:12px; border-radius:14px; border:1px solid #bbf7d0; background:#f0fdf4; color:#166534;">
          <b>Model loaded ✅</b> ({escape(str(MODEL_PATH.name))})
        </div>
        """

    return f"""
    <html>
      <head>
        <title>CIFAR-10 Classifier</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="font-family: Arial; background:#f8fafc; margin:0;">
        <div style="max-width: 980px; margin: 0 auto; padding: 28px 18px;">

          <!-- Header -->
          <div style="padding:18px; border-radius:16px; background:linear-gradient(90deg, #6d28d9, #ec4899); color:white;">
            <h2 style="margin:0;">CIFAR-10 Image Classifier</h2>
            <div style="opacity:0.95; margin-top:6px;">Upload an image → predict → view confidence bars</div>
          </div>

          <!-- Upload card -->
          <div style="margin-top:18px; padding:16px; background:#fff; border:1px solid #e5e7eb; border-radius:16px;">
            <h3 style="margin:0 0 10px 0; color:#111827;">Upload Image</h3>

            <form action="/predict" enctype="multipart/form-data" method="post"
                  style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
              <input name="file" type="file" accept="image/*" required
                     style="padding:10px; border:1px solid #d1d5db; border-radius:12px; background:#fff;" />
              <button type="submit"
                      style="padding:10px 14px; border-radius:12px; border:1px solid transparent;
                             background:#111827; color:#fff; cursor:pointer;">
                Predict
              </button>

              
            </form>


            {model_status}
            {error_box}
          </div>

          <!-- Result section (shows on home page) -->
          {result_html}

        </div>
      </body>
    </html>
    """


def render_result(image_b64: str, mime: str, filename: str, top_pred: str, topk: list) -> str:
    # Build top-k confidence bars
    rows = ""
    for idx, r in enumerate(topk):
        label = escape(r["label"])
        pct = float(r["probability"]) * 100
        badge = "Top" if idx == 0 else f"#{idx+1}"

        # Use slightly different gradients per rank (optional nice touch)
        if idx == 0:
            grad = "linear-gradient(90deg, #6d28d9, #ec4899)"
            badge_bg = "#6d28d9"
        elif idx == 1:
            grad = "linear-gradient(90deg, #2563eb, #06b6d4)"
            badge_bg = "#2563eb"
        else:
            grad = "linear-gradient(90deg, #16a34a, #84cc16)"
            badge_bg = "#16a34a"

        rows += f"""
        <div style="padding:12px; border:1px solid #e6e6e6; border-radius:12px; background:#fff; margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div style="font-weight:700; color:#1f2937;">
              {label}
              <span style="margin-left:8px; font-size:12px; font-weight:700; color:#fff; background:{badge_bg}; padding:3px 8px; border-radius:999px;">
                {badge}
              </span>
            </div>
            <div style="font-weight:700; color:#111827;">{pct:.2f}%</div>
          </div>

          <div style="margin-top:10px; height:12px; background:#f3f4f6; border-radius:999px; overflow:hidden;">
            <div style="height:12px; width:{pct:.2f}%; background:{grad}; border-radius:999px;"></div>
          </div>

          <div style="margin-top:6px; font-size:12px; color:#6b7280;">
            Confidence bar (0% → 100%)
          </div>
        </div>
        """

    return f"""
    <div style="margin-top:18px; display:grid; grid-template-columns: 1fr 1fr; gap:18px;">
      <!-- Left: image -->
      <div style="padding:16px; background:#fff; border:1px solid #e5e7eb; border-radius:16px;">
        <h3 style="margin:0 0 12px 0; color:#111827;">Uploaded Image</h3>
        <img src="data:{escape(mime)};base64,{image_b64}" alt="uploaded"
             style="width:100%; max-width:460px; border-radius:14px; border:1px solid #e5e7eb;" />
        <div style="margin-top:10px; color:#6b7280; font-size:13px;">File: {escape(filename)}</div>
      </div>

      <!-- Right: prediction -->
      <div style="padding:16px; background:#fff; border:1px solid #e5e7eb; border-radius:16px;">
        <h3 style="margin:0 0 12px 0; color:#111827;">Top Prediction</h3>

        <div style="padding:14px; border-radius:14px; border:1px solid #e5e7eb; background:#faf5ff;">
          <div style="font-size:20px; font-weight:800; color:#5b21b6;">
            {escape(top_pred)}
          </div>
          <div style="margin-top:6px; color:#6b7280; font-size:13px;">
            Predicted class from CIFAR-10 (10 categories)
          </div>
        </div>

        <h3 style="margin:18px 0 12px 0; color:#111827;">Confidence (Top-3)</h3>
        {rows}
      </div>
    </div>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Expected at: {MODEL_PATH}")
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        # Render home with a friendly message instead of plain JSON error
        return render_page(error_msg=f"Model not loaded. Expected at: {MODEL_PATH}")

    if not file.content_type or not file.content_type.startswith("image/"):
        return render_page(error_msg="Please upload a valid image file (jpg/png/webp).")

    # Read and decode image
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception:
        logger.exception("Failed to read image upload")
        return render_page(error_msg="Invalid or corrupted image file.")

    # Predict
    try:
        result = predictor.predict(image, top_k=3)
        top_pred = result["top_prediction"]["label"]
        topk = result["top_k"]

        # Base64 preview
        image_b64 = base64.b64encode(content).decode("utf-8")
        mime = file.content_type
        filename = file.filename or "uploaded_image"

        logger.info(f"Predicted={top_pred} file={filename}")

        result_html = render_result(
            image_b64=image_b64,
            mime=mime,
            filename=filename,
            top_pred=top_pred,
            topk=topk
        )

        # Show results on the same themed home page
        return render_page(result_html=result_html)

    except Exception:
        logger.exception("Prediction failed")
        return render_page(error_msg="Prediction failed due to a server error. Check logs/app.log.")
