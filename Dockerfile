
FROM python:3.10-slim
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
RUN python -m pip install --no-cache-dir -e .[dev]
CMD ["python", "-m", "pytest"]
