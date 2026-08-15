# Container-ready image for the sports-betting pipeline.
# Python 3.12 so BOTH stable TensorFlow and PyTorch install cleanly.
FROM python:3.12-slim

WORKDIR /app

# System deps for numpy/scipy/statsmodels wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-deep.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-deep.txt

COPY . .

# Verification gate
RUN python -m pytest tests/ -q

# Default: run the full pipeline (PoissonElo + ML + RL)
CMD ["python", "run_full_ml_rl.py"]
