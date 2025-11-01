FROM python:3.12.7-slim

WORKDIR backend

COPY requirements.txt .

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

# llm 폴더를 볼륨으로 추가
VOLUME ["/backend/llms"]

CMD ["python", "./server.py"]
