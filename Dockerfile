FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    openenv-core \
    pydantic \
    requests \
    openai \
    uvicorn \
    fastapi

COPY server/ ./server/
COPY pyproject.toml .

EXPOSE 7860

CMD ["python", "-c", "from server.app import main; main()"]
