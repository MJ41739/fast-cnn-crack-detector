# Production Deployment Plan - Crack Detection System

This document outlines the end-to-end architecture, optimization, and step-by-step configuration required to deploy the Crack Detection System professionally on the internet.

## User Review Required

> [!IMPORTANT]
> **Production API Routing (No CORS / No Mixed-Content)**
> Instead of hardcoding the Render/Railway backend URL in the React code, we will configure **Vercel Rewrites** in `vercel.json`. This routes `/api/:path*` to your public backend URL. The frontend javascript code will call `/api/predict` (relative path) in both development and production. This completely eliminates CORS issues and Mixed-Content blocks on mobile browsers.

> [!WARNING]
> **Render Free Tier Cold Starts and RAM Limits**
> - The Render Free Tier has a **512MB RAM limit**. Loading PyTorch (`torch`) + FastAPI into memory can exceed 300MB, pushing near the limit.
> - Render Free web services spin down (sleep) after 15 minutes of inactivity. The next request triggers a **50-80 second cold start** as the container starts and the machine learning model loads.
> - **Recommendation**: If budget permits, a $7/month Render Starter instance (2GB RAM, no sleeping) or **Railway Developer tier** is highly recommended for production ML web service hosting. For a 100% free tier with no cold-starts, **Hugging Face Spaces (Docker Space)** provides 16GB RAM and 2 vCPUs for free.

---

## Proposed Changes

We will introduce and modify several files across both the frontend and backend folders.

```
FASTCNN/
├── .github/
│   └── workflows/
│       └── deploy.yml               <-- [NEW] CI/CD Workflow
├── backend/
│   ├── main.py                     <-- [MODIFY] Add rate-limiting, /model-info, health validation
│   ├── requirements.txt            <-- [NEW] Slim backend dependencies
│   └── render.yaml                 <-- [NEW] Render infrastructure definition
├── frontend/
│   ├── vercel.json                 <-- [NEW] Vercel routing and edge proxying rules
│   └── .env.production             <-- [NEW] Production variables
├── Dockerfile                      <-- [NEW] Production multi-stage Docker build
└── docker-compose.prod.yml         <-- [NEW] Production local compose runner
```

---

### 1. Hosting Options Comparison

| Provider | Cost | CPU / RAM | GPU Support | Cold Start / Sleep | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Render (Free Web Service)** | Free | 0.1 vCPU / 512 MB | No | Yes (after 15m) | **Supported**, but cold starts are long and memory headroom is tight. |
| **Railway (Hobby Tier)** | $5/mo (free credits) | Shared / 8 GB | No | No | **Excellent** low-cost option. Easy setup, no cold starts. |
| **Hugging Face Spaces** | Free | 2 vCPU / 16 GB | Paid upgrade | No (sleeps after 48h) | **Best Free Option** for ML. High RAM and stable up-time. |
| **Google Cloud Run** | Pay-per-use | Configurable | Yes (Paid) | Scalable to 0 (Cold start ~3s) | **Best Enterprise Option**. Highly scalable. |

---

### 2. Implementation Details

#### [NEW] [vercel.json](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/frontend/vercel.json)
Configure Vercel to route frontend requests and proxy backend requests dynamically.
* We configure a rewrite rule mapping `/api/:path*` to the Render backend address.
* We set cache headers for static assets.

#### [NEW] [.env.production](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/frontend/.env.production)
Defines build time variables:
```env
VITE_API_URL=/api
```

#### [MODIFY] [main.py](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/backend/main.py)
* Integrate `slowapi` for API rate-limiting (e.g., max 10 requests per minute per IP for `/predict` to avoid spam).
* Implement payload validation: restrict uploads to `image/jpeg`, `image/png`, `image/bmp` and max size of `5MB`.
* Add the requested `GET /model-info` and alias `POST /batch-predict` endpoints.
* Implement startup pre-warming (run a dummy image through the model during startup so the first user does not experience latency).

#### [NEW] [requirements.txt](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/backend/requirements.txt)
Production python dependencies with fixed versions, avoiding heavy development dependencies:
```text
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic-settings>=2.2.1
torchvision>=0.17.0
onnxruntime>=1.17.0
pillow>=10.2.0
numpy>=1.26.0
python-multipart>=0.0.9
psutil>=5.9.8
slowapi>=0.1.9
```
*Note: We will request `cpu` specific wheels in the Docker environment to keep the image size small.*

#### [NEW] [Dockerfile](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/Dockerfile)
A highly optimized, multi-stage production Dockerfile:
* Stage 1: Build virtual environment and install `torch` CPU-only wheels (reducing image size from 3GB to ~700MB).
* Stage 2: Clean runner container copying only requirements and source code. Run under non-root user.

#### [NEW] [render.yaml](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/backend/render.yaml)
Render Blueprint configuration to deploy the backend service automatically.

#### [NEW] [.github/workflows/deploy.yml](file:///c:/Users/Mayur%20Jadhav/OneDrive/Desktop/FASTCNN/.github/workflows/deploy.yml)
GitHub Actions CI/CD Pipeline:
1. Runs code quality checks (flake8/eslint).
2. Runs validation tests.
3. Automatically deploys Frontend to Vercel (using Vercel CLI).
4. Triggers Render deploy webhook to rebuild and redeploy the backend Docker container.

---

## Verification Plan

### Automated Verification
* **Lint Checks**: ESLint for React, Flake8 for python.
* **Local Docker Test**: Run `docker compose -f docker-compose.prod.yml up` locally to verify that the container builds and starts up within 10 seconds.
* **Endpoints validation**: Test `/health`, `/model-info`, `/predict`, `/batch-predict` locally.

### Manual Verification
1. Access the Vercel staging deployment on a mobile device and laptop.
2. Confirm that the camera stream works, makes calls to the proxied `/api` endpoint, and successfully receives predictions.
3. Verify that uploading a non-image file or a file larger than 5MB returns a `400 Bad Request` or `413 Payload Too Large` error.
