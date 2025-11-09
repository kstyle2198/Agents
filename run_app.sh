#!/bin/bash

# 현재 스크립트의 경로로 이동 (어디서 실행해도 안정적)
cd "$(dirname "$0")"

# 백엔드 실행
echo "🚀 Starting backend..."
cd backend
python server.py &
cd ..

# 잠깐 대기 (서버 준비 시간)
sleep 3

# 프론트엔드 실행
echo "💻 Starting frontend..."
cd frontend
streamlit run app.py
