FROM python:3.11.11-slim@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.lock requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
CMD ["python", "-m", "pytest", "-q"]
