FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install pandas numpy scikit-learn joblib --no-cache-dir

ENTRYPOINT ["python", "src/predict_cli.py"]