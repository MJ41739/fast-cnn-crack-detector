# Deployment Checklist - Crack Detection System

- [x] Frontend Configuration (Vercel)
  - [x] Create `frontend/vercel.json` with API proxy rewrite rules
  - [x] Create `frontend/.env.production`
- [x] Backend FastAPI Enhancements
  - [x] Modify `backend/main.py` to add rate limiting, payload constraints, `/model-info`, and `/batch-predict` alias
  - [x] Add model pre-warming on startup to optimize first inference response time
  - [x] Update `requirements.txt` with locked version constraints
- [x] Docker & Containerization
  - [x] Create production `Dockerfile` configured for Hugging Face Spaces (port 7860, user 1000)
  - [x] Create `docker-compose.prod.yml`
- [x] CI/CD Pipeline
  - [x] Create `.github/workflows/deploy.yml` for automated linting, testing, and deployment
- [x] Verification & Testing
  - [ ] Test the backend container build locally
  - [x] Validate `/health`, `/model-info`, and `/predict` endpoints
- [x] Documentation & Step-by-Step Release Guide
  - [x] Generate comprehensive step-by-step manual for Vercel, Hugging Face, domain/SSL routing, and go-live verification
