FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
# The pip cache survives failed builds, so a re-run does not download finished packages again.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY detector ./detector
EXPOSE 8000
CMD ["uvicorn", "detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
