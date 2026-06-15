# Exact, offline-reproducible environment for the fxvol study.
FROM python:3.12-slim

WORKDIR /app

# System deps for numpy/scipy/arch wheels are bundled; keep the image lean.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e ".[dev]"

# Bring in data, config, tests, notebook and the dashboard.
COPY config.yaml ./
COPY data ./data
COPY tests ./tests
COPY notebooks ./notebooks
COPY app ./app

EXPOSE 8501

# Default: launch the interactive dashboard off the committed snapshot.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
