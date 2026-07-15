# Dockerfile

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY . .

RUN uv run --locked --no-dev python src/deco/download_datasets.py

RUN sed -i 's/\r$//' run_all.sh

RUN chmod +x run_all.sh

CMD ["./run_all.sh"]
