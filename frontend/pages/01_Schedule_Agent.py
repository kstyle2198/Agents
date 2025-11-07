import streamlit as st
import requests
import json
import time

# ==== 페이지 설정 ====
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")
st.title("Schedule Agent")
st.markdown("---")

API_URL = "http://localhost:8000/schedule"
CALENDAR_ID = "jongbaekim0710@gmail.com"

# ==== 채팅 기록 초기화 ====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_history" not in st.session_state:
    st.session_state.last_history = []

# ==== 대화 초기화 버튼 ====
def 대화초기화():
    col_reset, _ = st.columns([1, 5])
    with col_reset:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_history = []
            st.rerun()

# 채팅 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력
if prompt := st.chat_input("일정 입력 또는 브리핑을 요청해보세요."):
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 서버 요청
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ LangGraph 실행 중...")

        try:
            payload = {"user_input": prompt, "calendar_id": CALENDAR_ID}
            response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                history = data.get("history", [])
                st.session_state.last_history = history  # 오른쪽 패널에서 사용

                # LangGraph 실행 단계별 채팅 표시
                chat_text = ""
                for step in history:
                    step_msg = (
                        f"**Step: {step['node']}**\n"
                        f"```json\n{json.dumps(step['state'], ensure_ascii=False, indent=2)}\n```"
                    )
                    chat_text += step_msg + "\n\n"
                    message_placeholder.markdown(chat_text)
                    time.sleep(0.3)  # 스트리밍 느낌

                # 최종 결과
                chat_text = f"{data.get('result', '결과 없음')}"
                message_placeholder.markdown(chat_text)

                # 대화 기록 저장
                st.session_state.messages.append({"role": "assistant", "content": chat_text})

            else:
                error_msg = f"❌ 요청 실패 (HTTP {response.status_code})"
                message_placeholder.markdown(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"🚨 요청 중 오류 발생: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    


    with st.expander("Detailed Processes"):
        if st.session_state.last_history:
            for idx, step in enumerate(st.session_state.last_history, 1):
                with st.expander(f"Step {idx} — {step['node']}"):
                    st.json(step["state"])
        else:
            st.info("아직 실행 기록이 없습니다. 왼쪽에서 요청을 실행하세요.")
대화초기화()

