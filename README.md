# Urban Watch — Anomaly Detection API

An AI-powered service for detecting surface anomalies (cracks, rust, scratches, holes, etc.) in images using the **Grounded-SAM2** pipeline.

## 🚀 Getting Started

This project uses a Python virtual environment (`venv`) to manage dependencies.

### Prerequisites
- Python 3.9+
- macOS / Linux / Windows

### 1. Installation
Ensure your virtual environment is up to date with all necessary packages:
```bash
./venv/bin/pip install -r requirements.txt
```

### 2. Running the API
The API is built with FastAPI and runs on Uvicorn.

**Production Mode:**
```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Development Mode (Auto-reload):**
```bash
./venv/bin/uvicorn app.main:app --reload
```

### 3. Accessing the Dashboard (HTML)
The project includes a web-based dashboard for easy testing.
1. Start the API using one of the commands above.
2. Open your browser and go to: **[http://localhost:8000](http://localhost:8000)**
3. Drag and drop an image from the `assets/` folder to scan for anomalies.

---

## 🧪 Testing
We use `pytest` for automated quality assurance.

**Run all tests:**
```bash
./venv/bin/python -m pytest app/tests
```

**Run specific test file:**
```bash
./venv/bin/python -m pytest app/tests/test_detect.py
```

---

## 📂 Project Structure
- `app/main.py`: Entry point for the FastAPI application.
- `app/routes/`: API endpoint definitions (e.g., `/detect-anomalies`).
- `app/services/`: Core logic and model inference (Grounded-SAM2).
- `app/config/`: Configuration settings and model thresholds.
- `app/tests/`: Pytest suite and fixtures.
- `assets/`: Folder for your local test images.
- `outputs/`: Directory where processed images and masks are stored.

## 🛠 API Endpoints
- **GET `/health`**: Check if the service is running.
- **POST `/detect-anomalies`**: Upload an image to detect anomalies.
- **GET `/docs`**: Interactive Swagger documentation.

## ⚠️ Configuration
You can adjust detection thresholds and upload limits in `app/config/settings.py`:
- `MAX_UPLOAD_BYTES`: Maximum allowed image size (default 20MB).
- `BOX_THRESHOLD`: Sensitivity of the detection boxes.
- `TEXT_THRESHOLD`: Sensitivity of the label matching.
