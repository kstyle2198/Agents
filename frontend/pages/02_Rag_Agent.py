import os
import streamlit as st
import requests
import json
import uuid
from typing import Dict, Any

from utils.my_stt import AudioTranscriber # 주석 처리 또는 적절히 대체
transcriber = AudioTranscriber()

# ---------------------------------------------------------
# 1. 환경 설정
# ---------------------------------------------------------
# FastAPI 서버의 주소와 엔드포인트 URL을 정확히 설정하세요.
# (FastAPI 서버가 8000번 포트에서 실행 중이라고 가정)
API_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_STREAM_ENDPOINT = f"{API_BASE_URL}/rag_chat_stream"

st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------
# 2. 세션 상태 초기화
# ---------------------------------------------------------
if "messages" not in st.session_state:
    # 대화 이력: [{"role": "user/assistant", "content": "..."}]
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # 멀티턴 유지를 위한 세션 ID (없으면 새로 생성)
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages.append(
        {"role": "assistant", "content": "안녕하세요! 질문을 입력해주세요."}
    )

# ---------------------------------------------------------
# 3. UI 구성
# ---------------------------------------------------------
st.title("Rag Agent")
st.caption(f"현재 세션 ID: `{st.session_state.session_id}`")

# 대화 이력 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------------------------------------
# 4. 사용자 입력 및 스트리밍 처리 함수
# ---------------------------------------------------------
def process_streaming_response(data: Dict[str, Any]):
    """
    FastAPI 서버에 요청을 보내고 스트리밍 응답을 처리합니다.
    """
    try:
        # API 호출 (stream=True 설정이 중요)
        response = requests.post(API_STREAM_ENDPOINT, json=data, stream=True, timeout=120)
        response.raise_for_status() # HTTP 오류 시 예외 발생

        # AI 응답을 표시할 빈 placeholder 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 스트림 응답 처리 (JSON Lines 파싱)
            for line in response.iter_lines():
                if line:
                    try:
                        # 라인을 UTF-8로 디코딩하고 JSON으로 파싱
                        json_line = json.loads(line.decode('utf-8'))
                        
                        if json_line.get("type") == "token":
                            # 토큰을 누적하여 표시
                            token = json_line.get("content", "")
                            full_response += token
                            # '▌'은 현재 토큰이 입력되고 있음을 보여주는 커서 효과
                            message_placeholder.markdown(full_response + "▌") 
                        
                        elif json_line.get("type") == "end":
                            # 스트림 종료 시 최종 세션 ID를 캡처하여 업데이트
                            final_session_id = json_line.get("session_id")
                            if final_session_id:
                                st.session_state.session_id = final_session_id
                                st.caption(f"현재 세션 ID: `{st.session_state.session_id}`")
                            break # 스트림 종료

                    except json.JSONDecodeError:
                        # 유효하지 않은 JSON 라인 건너뛰기
                        continue 

            # 최종 응답 표시 및 대화 이력 업데이트
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except requests.exceptions.RequestException as e:
        error_message = f"🚨 API 통신 오류: FastAPI 서버가 실행 중인지 확인하세요. (오류: {e})"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})


# ---------------------------------------------------------
# 5. 입력 처리
# ---------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지를 이력에 추가하고 표시
    # if prompt.audio:
    #     audio_bytes = prompt.audio.read()
    #     prompt = transcriber.transcribe_audio2(audio_bytes=audio_bytes)
    # else: pass
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # API 요청 데이터 준비
    api_request_data = {
        "question": prompt,
        "session_id": st.session_state.session_id
    }

    process_streaming_response(api_request_data)


# --- Agent 작업 과정 및 Chat History ---
if st.session_state.session_id:
    response = requests.get(f"{API_BASE_URL}/rag_threads", params={"thread_id": st.session_state.session_id})
    if response.status_code == 200:
        threads_history = response.json().get("threads", [])
        st.session_state.threads = list(reversed(threads_history))

    

    with st.expander(f"🔍 ElasticSearch 검색 결과 - {st.session_state.session_id}"):
        if st.session_state.threads:
            for h in st.session_state.threads:
                try:
                    if h[1][0]  == "search":
                        with st.container(border=True, height=600):
                            for i in h[-2]:
                                for k in i[6]["search_results"]:
                                    st.warning(f"{k["lv1_cat"]}/{k["lv2_cat"]}/{k["lv3_cat"]}/{k["lv4_cat"]}/{k["filename"]}({k["page"]} page) -  score: {k["_score"]}")
                                    st.info(k["page_content"])
                                    # st.info(k)
                except: pass

    with st.expander(f"🔍 Rag Agent 작업 과정 보기 - {st.session_state.session_id}"):
        if st.session_state.threads:
            for h in st.session_state.threads:
                if h[1] != [] and h:
                    with st.container(border=True, height=150):
                        st.warning(h[1])
                        st.info(h[-2])

    with st.expander(f"🧾 Chat History - {st.session_state.session_id}"):
        for m in st.session_state.messages:
            st.info(m)


# ---------------------------------------------------------
# 6. 추가 기능: 세션 초기화
# ---------------------------------------------------------
if st.button("🗑️ 대화 초기화"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "새로운 세션이 시작되었습니다. 무엇을 도와드릴까요?"}]
    st.rerun() # 앱 재실행하여 변경된 상태 반영


import os
from io import BytesIO
import markdown2
from xhtml2pdf import pisa

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


# -------------------------------------------------
# Markdown → HTML → PDF (한글/특수문자 정상)
# -------------------------------------------------
def create_markdown_pdf(messages):
    buffer = BytesIO()

    # 1️⃣ 한국어 CIDFont 등록 (🔥 핵심)
    pdfmetrics.registerFont(
        UnicodeCIDFont("HYSMyeongJo-Medium")
    )

    # 2️⃣ 메시지 → Markdown
    md_text = ""
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        md_text += f"### {role}\n\n{content}\n\n---\n\n"

    # 3️⃣ Markdown → HTML
    html_body = markdown2.markdown(
        md_text,
        extras=[
            "tables",
            "fenced-code-blocks",
            "strike",
            "task_list",
            "code-friendly"
        ]
    )

    # 4️⃣ HTML + CSS
    html = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            body {{
                font-family: HYSMyeongJo-Medium;
                font-size: 10pt;
                line-height: 1.6;
                padding: 20px;
            }}

            h1, h2, h3 {{
                font-family: HYSMyeongJo-Medium;
                margin-top: 15px;
            }}

            pre, code {{
                font-family: HYSMyeongJo-Medium;
                background: #f4f4f4;
                font-size: 8pt;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
            }}

            th, td {{
                border: 1px solid #666;
                padding: 6px;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # 5️⃣ PDF 생성
    pisa_status = pisa.CreatePDF(
        html,
        dest=buffer,
        encoding="utf-8"
    )

    if pisa_status.err:
        raise RuntimeError("PDF 생성 실패")

    buffer.seek(0)
    return buffer


# ------------------------------
# PDF 다운로드 버튼 (Sidebar)
# ------------------------------
with st.sidebar:
    st.subheader("📄 PDF Export (Markdown 스타일 유지)")
    file_name = st.text_input("PDF filename", value="testfile")

    if st.button("💾 PDF 생성"):
        pdf_buffer = create_markdown_pdf(st.session_state.messages[-2:])
        st.download_button(
            label="📥 PDF 다운로드",
            data=pdf_buffer,
            file_name=f"{file_name}.pdf",
            mime="application/pdf"
        )

with st.sidebar:
    st.markdown("""
- Attention Is All You Need (Vaswani et al., 2017)
- Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- From Local to Global: A Graph RAG Approach to Query-Focused Summarization(Edge et al., 2024)
- Visual Instruction Tuning (Liu et al., 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)
- REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory (Hu et al., 2022)
                """)
    

