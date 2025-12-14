import os
import streamlit as st
import requests
import pandas as pd

API_BASE = os.getenv("BASE_URL", "http://localhost:8000")   # FastAPI 주소

st.set_page_config(page_title="Chat History Admin", layout="wide")
st.title("💬 Chat History 관리")
columns = ['id', 'session_id', 'query', 'refined_query', 'answer']

# -------------------------
# 입력 영역
# -------------------------
with st.sidebar:
    st.header("조회 설정")
    table_name = st.text_input("Table Name", value="agent")
    limit = st.number_input("Limit", min_value=1, max_value=500, value=50)
    id_column = st.text_input("ID Column", value="session_id")

    fetch_btn = st.button("📥 채팅 히스토리 조회")

    st.markdown("---")
    session_id = st.text_input("Session ID")
    if st.button("Data 삭제"):
        
        try:
            response = requests.post(
                f"{API_BASE}/delete_chat_by_id",
                params={
                    "table_name": table_name,
                    "id_column": id_column,
                    "record_id": session_id
                },
                timeout=10
            )

            if response.status_code == 200:
                st.success("✅ 채팅 기록이 성공적으로 삭제되었습니다.")
                st.json(response.json())
                st.rerun()
            else:
                st.error(f"❌ 삭제 실패 (status: {response.status_code})")
                st.text(response.text)
        except Exception as e:
            st.error(f"서버 요청 중 오류 발생: {e}")


# -------------------------
# 데이터 조회
# -------------------------

if fetch_btn:
    try:
        resp = requests.get(
            f"{API_BASE}/chat_history",
            params={
                "table_name": table_name,
                "limit": limit
            }
        )
        resp.raise_for_status()
        data = resp.json()["chat_history"]

        st.session_state.chat_history = data

    except Exception as e:
        st.error(f"조회 실패: {e}")

try:
    for doc in list(reversed(st.session_state.chat_history)):
        with st.expander(f"Session ID: {doc[1]} - Query: {doc[2]}"):
            st.warning(doc[-1])
except Exception as e: 
    st.error("사이드바의 조회버튼을 클릭하세요")

