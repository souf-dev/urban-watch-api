# Urban Watch - Anomaly Detection API

FastAPI service for detecting surface anomalies in images using:

1. **GroundingDINO Swin-T** to find anomaly boxes from text prompts.
2. **SAM2.1 Large** to turn those boxes into segmentation masks.

The API endpoint is:

```text
POST /detect-anomalies
```

The dashboard is available at:

```text
http://localhost:8000
```

## Important

The real model setup uses about **1.6 GB** of checkpoint files:

```text
checkpoints/groundingdino_swint_ogc.pth
checkpoints/sam2.1_hiera_large.pt
```

Do not commit these files to GitHub. The repo ignores `checkpoints/`.

## Recommended Setup: Docker GPU

Use this for real inference and deployment.

Requirements:

- Docker
- Linux machine with NVIDIA GPU
- NVIDIA Container Toolkit installed

Mac can build parts of the project, but it will not run NVIDIA GPU inference.

### 1. Download Model Weights

From the project root:

```bash
mkdir -p checkpoints outputs

curl -L -o checkpoints/groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

curl -L -o checkpoints/sam2.1_hiera_large.pt https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
```

Expected files:

```text
checkpoints/groundingdino_swint_ogc.pth  ~694 MB
checkpoints/sam2.1_hiera_large.pt       ~898 MB
```

### 2. Build Docker Image

```bash
docker compose -f docker-compose.gpu.yml build
```

This installs:

- Python 3.10
- Torch 2.5.1
- Torchvision 0.20.1
- GroundingDINO
- SAM2
- FastAPI app dependencies

### 3. Start API

```bash
docker compose -f docker-compose.gpu.yml up
```

Open:

```text
http://localhost:8000
```

Stop with:

```bash
Ctrl+C
```

Or run in background:

```bash
docker compose -f docker-compose.gpu.yml up -d
```

Stop background container:

```bash
docker compose -f docker-compose.gpu.yml down
```

## Local Python Setup

Use this only for development, editing, and basic checks. Full SAM2 latest is cleaner in Docker/Linux GPU.

### 1. Create Python 3.10 Environment

```bash
/usr/local/bin/python3.10 -m venv .venv310
source .venv310/bin/activate
python --version
```

Expected:

```text
Python 3.10.x
```

### 2. Install Base Requirements

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Install GroundingDINO

```bash
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
```

### 4. Install SAM2 Local Workaround

On this Mac, pip may not provide `torch>=2.5.1`. If that happens, install SAM2 without dependency resolution:

```bash
pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/sam2.git
pip install hydra-core iopath tqdm
pip install --force-reinstall "numpy<2" "opencv-python==4.10.0.84"
```

Check imports:

```bash
python -c "import cv2, numpy, torch, groundingdino, sam2; print('OK', numpy.__version__, cv2.__version__, torch.__version__)"
```

## Running Without Docker

If local dependencies and checkpoints are installed:

```bash
source .venv310/bin/activate
uvicorn app.main:app --reload
```

Then open:

```text
http://localhost:8000
```

## Configuration

Model paths are configured in `app/config/settings.py`.

Defaults:

```text
GROUNDING_DINO_CHECKPOINT=checkpoints/groundingdino_swint_ogc.pth
SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_large.pt
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
```

You can override them with environment variables:

```bash
GROUNDING_DINO_CHECKPOINT=/models/groundingdino_swint_ogc.pth \
SAM2_CHECKPOINT=/models/sam2.1_hiera_large.pt \
uvicorn app.main:app --reload
```

Detection thresholds:

```text
BOX_THRESHOLD=0.30
TEXT_THRESHOLD=0.25
```

Edit them in:

```text
app/config/settings.py
```

## API Endpoints

```text
GET  /health
POST /detect-anomalies
GET  /docs
GET  /
```

Upload field for `/detect-anomalies`:

```text
image
```

Response includes:

```text
detections[].label
detections[].confidence
detections[].box
detections[].color
final_image
```

## Tests

```bash
source .venv310/bin/activate
python -m pytest app/tests
```

## Project Structure

```text
app/main.py                  FastAPI app
app/routes/detect.py         Upload endpoint
app/services/model_service.py GroundingDINO + SAM2 inference
app/utils/image_utils.py     Draw and save masks/results
app/config/settings.py       Paths, prompts, thresholds
checkpoints/                 Local model weights, ignored by git
outputs/                     Generated result images, ignored by git
Dockerfile.gpu               GPU Docker image
docker-compose.gpu.yml       GPU Docker runner
requirements-docker.txt      Docker Python dependencies
```

## Troubleshooting

### Pyrefly cannot find `sam2` or `groundingdino`

Set your IDE interpreter to:

```text
.venv310/bin/python
```

The repo also includes stubs under `stubs/` for static analysis.

### Docker says no GPU available

You need a Linux NVIDIA GPU machine with NVIDIA Container Toolkit. Mac Docker does not provide NVIDIA GPU access.

### Missing checkpoint error

Make sure these exist:

```text
checkpoints/groundingdino_swint_ogc.pth
checkpoints/sam2.1_hiera_large.pt
```
