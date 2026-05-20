FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY README.md LICENSE ./

RUN pip install --no-cache-dir -e .

HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=2 \
    CMD python -c "from prompt_trace import Tracer; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "prompt_trace.cli"]
CMD ["--help"]
