import streamlit as st
import requests
import pandas as pd
import uuid
import json
from dotenv import load_dotenv
load_dotenv()

# --- 설정 ---
# 로컬에서 FastAPI 서버를 실행하는 경우의 주소입니다.
# 만약 다른 주소에서 서버를 실행한다면 이 값을 변경해주세요.
FASTAPI_URL = "http://127.0.0.1:8000/astream"

# 데이터베이스에 존재하는 테이블 목록을 여기에 정의합니다.
# 사용자가 이 목록에서 테이블을 선택하게 됩니다.
AVAILABLE_TABLES = ["ship_fuel_efficiency", "builder"]

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")


st.title("SQL Agent")
st.caption("자연어 질문을 SQL 쿼리로 변환하고 실행 결과를 보여줍니다.")

# --- 세션 상태 초기화 ---
# 세션 ID가 없으면 새로 생성하여 고유한 대화를 유지합니다.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 대화 내용을 저장할 리스트를 초기화합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []


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
    st.markdown(f"**현재 세션 ID:** `{st.session_state.session_id}`")

    if st.button("🔄 New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# --- 메인 화면 ---

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 메시지 내용에 따라 다르게 표시
        if "dataframe" in message["content"]:
            st.markdown(message["content"]["text"])
            # st.dataframe(pd.read_json(message["content"]["dataframe"]))
        elif "error" in message["content"]:
            st.error(message["content"]["error"])
        else:
            st.markdown(message["content"])

# 사용자 질문 입력
if prompt := st.chat_input("데이터에 대해 질문해보세요..."):
    # 1. 사용자 질문을 대화 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 테이블 선택 유효성 검사
    if not selected_tables:
        st.error("분석할 테이블을 사이드바에서 하나 이상 선택해주세요.")
        st.stop()

    # 3. FastAPI 백엔드에 보낼 요청 데이터 구성
    request_data = {
        "question": prompt,
        "target_tables": selected_tables,
        "session_id": st.session_state.session_id,
        "attempts": 0, # 초기 시도는 0
    }

    # 4. API 요청 및 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("Agent가 생각 중입니다... 🤔"):
            try:
                # 스트리밍 엔드포인트에 POST 요청
                response = requests.post(FASTAPI_URL, json=request_data, timeout=120)
                response.raise_for_status()  # 200번대 상태 코드가 아니면 예외 발생

                response_data = response.json()
                final_output = response_data.get("final_output", {})
                history = response_data.get("history", [])

                # Agent의 답변 표시
                st.markdown(final_output.get("query_result", "결과를 요약하는 데 실패했습니다."))
                
                # SQL 쿼리 표시
                if final_output.get("sql_query"):
                    st.code(final_output["sql_query"], language="sql")

                # 쿼리 실행 결과(DataFrame) 표시
                query_rows = final_output.get("query_rows", [])
                if query_rows:
                    df = pd.DataFrame(query_rows)
                    st.dataframe(df)
                    # 대화 기록에 저장하기 위해 dataframe을 json으로 변환
                    df_json = df.to_json(orient='split')
                    assistant_response = {
                        "text": final_output.get("query_result", ""),
                        "dataframe": df_json
                    }
                else:
                    assistant_response = final_output.get("query_result", "결과가 없습니다.")

                # Agent 작업 과정(History) 표시
                with st.expander("🔍 Agent 작업 과정 보기"):
                    reverse_history = list(reversed(history))
                    for h in reverse_history:
                        st.warning(h[1])
                        st.info(h[-2])
                    # st.json(reverse_history)

                # 대화 기록에 Agent 응답 추가
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                    )

            except requests.exceptions.RequestException as e:
                # 네트워크 또는 HTTP 오류 처리
                error_message = f"API 요청 중 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": {"error": error_message}})
            except Exception as e:
                # 기타 예외 처리
                error_message = f"처리 중 알 수 없는 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": {"error": error_message}})


    