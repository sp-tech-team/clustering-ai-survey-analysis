FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user:user . /app
RUN mkdir -p /app/cache /app/db && chown -R user:user /app
USER user

CMD ["gunicorn", "app:server", "--workers", "4", "--bind", "0.0.0.0:7860"]