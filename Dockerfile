FROM python:3.9.23-bookworm

WORKDIR /home/API_GATEWAY

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]