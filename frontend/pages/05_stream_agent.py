import json
import time
import uuid
import requests
import streamlit as st
import httpx
import os
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")
st.title("💬 Streaming Chat Interface")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "threads" not in st.session_state:
    st.session_state.threads = []

if "search_urls" not in st.session_state:
    st.session_state.search_urls = []


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

        def stream_from_api():
            with httpx.stream(
                "POST",
                f"{BASE_URL}/chat",
                json={"query": prompt, "thread_id": st.session_state.thread_id},
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

        full_response = placeholder.write_stream(stream_from_api())

    st.session_state.messages.append({"role": "assistant", "content": full_response})



if "search_urls" in st.session_state and st.session_state.search_urls:
    urls = st.session_state.search_urls
    num_cols = 3

    st.markdown("### 🔗 Related Research Links")

    # 3개씩 나누어 행 단위로 표시
    for i in range(0, len(urls), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            index = i + j
            if index < len(urls):
                with col:
                    with st.container(border=True, height=50):
                        st.write(urls[index])
else:
    st.info("검색 결과가 없습니다.")


# --- Agent 작업 과정 및 Chat History ---
if st.session_state.thread_id:
    response = requests.get(f"{BASE_URL}/threads", params={"thread_id": st.session_state.thread_id})
    if response.status_code == 200:
        threads_history = response.json().get("threads", [])
        st.session_state.threads = list(reversed(threads_history))

    with st.expander(f"🔍 Agent 작업 과정 보기 - {st.session_state.thread_id}"):
        if st.session_state.threads:
            for h in st.session_state.threads:
                if h[1] != []:
                    with st.container(border=True, height=150):
                        st.warning(h[1])
                        st.info(h[-2])

    with st.expander(f"🧾 Chat History - {st.session_state.thread_id}"):
        for m in st.session_state.messages:
            st.info(m)


# --- 초기화 버튼 ---
if st.button("🗑️ 대화 초기화"):
    st.session_state.messages = []
    st.session_state.search_urls = []
    st.session_state.thread_id = None
    st.session_state.threads = None
    st.rerun()
