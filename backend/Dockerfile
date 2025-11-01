FROM python:3.12.7-slim

WORKDIR backend

COPY requirements.txt .

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./server.py"]
