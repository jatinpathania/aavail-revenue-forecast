# AAVAIL Revenue Forecast | Docker Image
FROM python:3.12-slim

LABEL title="AAVAIL Revenue Forecast" \
      description="Next-30-day revenue prediction API for AAVAIL"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ingest the data and train the production models at build time,
# then run the full unit test suite so a broken image fails to build
RUN python cslib.py --clean \
 && python model.py \
 && python run-tests.py

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "600", "app:app"]
