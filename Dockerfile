# Imagen única: front web + CLI + PySpark (Java incluido). Debian bookworm trae OpenJDK 17.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
 && rm -rf /var/lib/apt/lists/*
# sin JAVA_HOME fijo: PySpark usa el `java` del PATH (funciona en amd64 y arm64)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY data ./data

EXPOSE 8000
CMD ["python", "-m", "src.web", "--host", "0.0.0.0", "--port", "8000"]
