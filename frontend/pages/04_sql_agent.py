import json
import time
import uuid
import requests
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")
st.title("SQL Agent")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "threads" not in st.session_state:
    st.session_state.threads = []

AVAILABLE_TABLES = ["ship_fuel_efficiency"]
# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    # 사용자가 분석할 테이블을 선택하는 멀티셀렉트 위젯
    selected_tables = st.pills(
        "분석할 테이블을 선택하세요:",
        options=AVAILABLE_TABLES,
        default=AVAILABLE_TABLES[0] if AVAILABLE_TABLES else None, selection_mode="multi",
        )
    st.info("선택된 테이블을 기반으로 SQL 쿼리를 생성합니다.")
    st.markdown(f"**현재 Thread ID:** `{st.session_state.thread_id}`")


def clean_and_parse_sse_line(line: str) -> str:
    """Extracts the content from an SSE 'data:' line"""
    if line.startswith("data:"):
        return line[len("data:"):].strip()
    return ""


# --- 채팅 히스토리 표시 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- 사용자 입력 (st.chat_input 사용) ---
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 스레드 초기화
    if not st.session_state.thread_id:
        st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    search_urls = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for i in range(3):
            placeholder.markdown("🤔 Assistant is thinking" + "." * (i + 1))
            time.sleep(0.4)

        def stream_from_api(request_data):
            with httpx.stream(
                "POST",
                f"{API_URL}/astream",
                json=request_data,
                timeout=60.0,
                ) as response:
                    for chunk in response.iter_text():
                        if not chunk.strip():
                            continue
                        try:
                            data = json.loads(chunk.replace("data: ", "").strip())
                        except json.JSONDecodeError:
                            continue

                        if data.get("type") == "content":
                            yield data.get("content", "")
                        elif data.get("type") == "search_results":
                            st.session_state.search_urls.extend(data.get("urls", []))

        request_data = {
            "question": prompt,
            "target_tables": selected_tables,
            "thread_id": st.session_state.thread_id,
            "attempts": 0, # 초기 시도는 0
            }
        
        full_response = placeholder.write_stream(stream_from_api(request_data))

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Agent 작업 과정 및 Chat History ---
if st.session_state.thread_id:
    response = requests.get(f"{API_URL}/sql_threads", params={"thread_id": st.session_state.thread_id})
    if response.status_code == 200:
        threads_history = response.json().get("threads", [])
        st.session_state.threads = list(reversed(threads_history))

    with st.expander(f"🔍 SQL Agent 작업 과정 보기 - {st.session_state.thread_id}"):
        if st.session_state.threads:
            for h in st.session_state.threads:
                if h[1] != [] and h:
                    with st.container(border=True, height=150):
                        st.warning(h[1])
                        st.info(h[-2])

    with st.expander(f"🧾 Chat History - {st.session_state.thread_id}"):
        for m in st.session_state.messages:
            st.info(m)

# --- 초기화 버튼 ---
if st.button("🗑️ 대화 초기화"):
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.threads = None
    st.rerun()
