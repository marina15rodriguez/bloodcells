"""FastAPI web interface for blood cell classification.

Endpoints:
    GET  /          — drag-and-drop HTML frontend
    GET  /health    — health check
    POST /predict   — accepts an image file, returns classification result

Usage:
    cd api
    uvicorn main:app --reload --port 8000
    open http://localhost:8000
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from predict import load_model, run_prediction          # when run from api/
except ImportError:
    from api.predict import load_model, run_prediction      # when run from project root (Docker)


CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    str(Path(__file__).resolve().parent.parent / "results" / "best_model.pth")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(CHECKPOINT)
    yield


app = FastAPI(title="Blood Cell Classification", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        result = run_prediction(image_bytes)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Blood Cell Classification</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117; color: #e0e0e0;
      min-height: 100vh; display: flex; flex-direction: column;
      align-items: center; padding: 40px 20px;
    }
    h1 { font-size: 1.8rem; font-weight: 600; margin-bottom: 6px; color: #fff; }
    .subtitle { font-size: 0.95rem; color: #888; margin-bottom: 36px; text-align: center; }

    .drop-zone {
      width: 100%; max-width: 520px;
      border: 2px dashed #444; border-radius: 14px;
      padding: 48px 24px; text-align: center; cursor: pointer;
      transition: border-color 0.2s, background 0.2s; background: #181c27;
    }
    .drop-zone:hover, .drop-zone.drag-over { border-color: #5b8dee; background: #1c2237; }
    .drop-zone p { color: #aaa; font-size: 0.95rem; margin-top: 10px; }
    .drop-zone .icon { font-size: 2.5rem; }
    #file-input { display: none; }

    .btn {
      margin-top: 20px; padding: 12px 32px; background: #5b8dee;
      color: white; border: none; border-radius: 8px; font-size: 1rem;
      cursor: pointer; transition: background 0.2s;
    }
    .btn:hover { background: #4a7de0; }
    .btn:disabled { background: #333; color: #666; cursor: not-allowed; }

    #status { margin-top: 20px; font-size: 0.9rem; color: #aaa; min-height: 22px; }

    #results { margin-top: 32px; width: 100%; max-width: 700px; display: none; }

    .result-box {
      background: #181c27; border-radius: 12px; padding: 24px;
      display: flex; gap: 28px; align-items: flex-start;
    }
    #preview { width: 180px; height: 180px; border-radius: 8px; object-fit: cover; flex-shrink: 0; }

    .info { flex: 1; }
    .predicted-class {
      font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 4px;
    }
    .confidence { font-size: 0.9rem; color: #888; margin-bottom: 20px; }

    .bar-label { font-size: 0.82rem; color: #aaa; margin-bottom: 4px; }
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .bar-name { width: 100px; font-size: 0.82rem; color: #ccc; flex-shrink: 0; }
    .bar-track { flex: 1; background: #2a2f45; border-radius: 4px; height: 10px; }
    .bar-fill { height: 10px; border-radius: 4px; background: #5b8dee; transition: width 0.4s; }
    .bar-pct { width: 44px; font-size: 0.8rem; color: #aaa; text-align: right; flex-shrink: 0; }

    .error {
      color: #ff6b6b; background: #2a1a1a; border-radius: 8px;
      padding: 12px 18px; margin-top: 16px; font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <h1>Blood Cell Classification</h1>
  <p class="subtitle">Upload a microscopy image — the model identifies the white blood cell type.</p>

  <div class="drop-zone" id="drop-zone">
    <div class="icon">🔬</div>
    <p>Drag &amp; drop a blood cell image here</p>
    <p>or click to browse &nbsp;·&nbsp; JPEG / PNG</p>
    <input type="file" id="file-input" accept=".jpg,.jpeg,.png"/>
  </div>

  <button class="btn" id="run-btn" disabled>Classify</button>
  <div id="status"></div>

  <div id="results">
    <div class="result-box">
      <img id="preview" alt="Uploaded image"/>
      <div class="info">
        <div class="predicted-class" id="pred-class"></div>
        <div class="confidence" id="pred-conf"></div>
        <div class="bar-label">Class probabilities</div>
        <div id="bars"></div>
      </div>
    </div>
  </div>

  <script>
    const dropZone  = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const runBtn    = document.getElementById('run-btn');
    const status    = document.getElementById('status');
    const results   = document.getElementById('results');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) setFile(f); else showError('Please drop an image file.');
    });
    fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

    function setFile(f) {
      selectedFile = f;
      dropZone.querySelector('p').textContent = `Selected: ${f.name}`;
      runBtn.disabled = false;
      results.style.display = 'none';
      status.textContent = '';
    }

    runBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      runBtn.disabled = true;
      status.textContent = 'Classifying…';
      results.style.display = 'none';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const resp = await fetch('/predict', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) { showError(data.detail || 'Server error'); return; }

        document.getElementById('preview').src = 'data:image/png;base64,' + data.original_png;
        document.getElementById('pred-class').textContent = data.predicted_class;
        document.getElementById('pred-conf').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

        const bars = document.getElementById('bars');
        bars.innerHTML = '';
        const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([cls, pct]) => {
          bars.innerHTML += `
            <div class="bar-row">
              <div class="bar-name">${cls}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${(pct*100).toFixed(1)}%"></div></div>
              <div class="bar-pct">${(pct*100).toFixed(1)}%</div>
            </div>`;
        });

        results.style.display = 'block';
        status.textContent = 'Done.';
      } catch (err) {
        showError('Request failed: ' + err.message);
      } finally {
        runBtn.disabled = false;
      }
    });

    function showError(msg) {
      status.innerHTML = `<div class="error">${msg}</div>`;
    }
  </script>
</body>
</html>
"""
