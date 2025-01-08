FROM python:3.9-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir numpy==1.26.4
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

WORKDIR /app
COPY .env .env
COPY . /app


RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "gpt_openai.py", "--server.address=0.0.0.0"]
