# Stage 1: Build dependencies in a temporary container
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheels first to optimize image size
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install other backend dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight runner container
FROM python:3.12-slim AS runner

WORKDIR /app

# Install system dependencies for imaging libraries (OpenCV/PIL support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000) for Hugging Face Spaces sandbox compatibility
RUN useradd -m -u 1000 user

# Copy installed Python packages from builder to user's home directory with correct ownership
COPY --from=builder --chown=user:user /root/.local /home/user/.local

# Copy backend codebase and weights into container with correct user permissions
COPY --chown=user:user backend/ /app/backend/
COPY --chown=user:user models/ /app/models/
COPY --chown=user:user utils/ /app/utils/
COPY --chown=user:user checkpoints/ /app/checkpoints/

# Grant write access to /app for the non-root user
RUN chown -R user:user /app

USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Default MLOps environment configurations
ENV DEVICE=cpu
ENV PORT=7860
ENV HOST=0.0.0.0
ENV ACTIVE_MODEL_NAME=onnx_quantized

# Expose Hugging Face default port
EXPOSE 7860

# Launch Uvicorn ASGI Server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
