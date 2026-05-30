---
title: Crack Detection API
emoji: 🧱
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Crack Detection API (Hugging Face Space)

This space hosts the backend API for the Concrete Crack Detection system. It runs a FastAPI server inside a Docker container.

### API Endpoints
*   `GET /health` - Service health status
*   `GET /model-info` - Active model specifications
*   `POST /predict` - Single image classification (with probability output)
*   `POST /batch-predict` - Rapid batched scanning
