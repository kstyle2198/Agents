import streamlit as st
import requests
import json
import time
import os
from pathlib import Path

from utils.my_stt import AudioTranscriber # 주석 처리 또는 적절히 대체

# ==== 페이지 설정 ====
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")


BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_URL = f"{BASE_URL}/schedule"
CALENDAR_ID = "jongbaekim0710@gmail.com"

# ==== 채팅 기록 및 상태 초기화 ====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_history" not in st.session_state:
    st.session_state.last_history = []
# 사용자 입력 (prompt)을 세션 상태에 저장하여 main()과 fixed_input_bar() 간에 전달
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = None

# ==== 대화 초기화 버튼 ====
def 대화초기화():
    # 이 부분은 페이지 상단에 위치하여야 스크롤에 영향을 받지 않으므로 main() 함수 바깥에 배치
    col_reset, _ = st.columns([1, 7])
    with col_reset:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_history = []
            st.session_state.user_prompt = None # prompt도 초기화
            st.rerun()
            
# 대화초기화() # 메인 제목 아래에 바로 렌더링

# ==== 음성 입력 버튼 (더미 함수로 대체) ==== 
def 음성입력():
    transcriber = AudioTranscriber()
    st.info("🎙️ 음성 녹음 중...(5초)")
    text = transcriber.run()  # 녹음 + STT
    st.success("✅ 음성 인식 완료!")
    return text

# **[핵심 변경] 하단 고정 입력창 함수**
def fixed_input_bar():
    # 챗봇 입력창을 페이지 하단에 고정하는 CSS를 주입
    st.markdown(
        """
        <style>
        /* 하단 고정 영역을 위한 컨테이너 스타일 */
        .fixed-footer-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            /* Streamlit의 메인 콘텐츠 영역과 동일한 최대 너비를 사용 */
            max-width: 730px; 
            margin: 0 auto;
            padding: 10px 0;
            background-color: white; /* 입력창 뒤 배경색 설정 */
            border-top: 1px solid #e6e6e6; /* 상단 구분선 */
            z-index: 100;
        }

        /* Streamlit이 생성하는 chat_input의 상위 폼 컨테이너를 숨김 */
        .stForm {
            margin: 0;
            padding: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 하단 고정 컨테이너 시작
    with st.container():
        # CSS 스타일을 적용할 HTML 요소를 직접 렌더링
        st.markdown('<div class="fixed-footer-container">', unsafe_allow_html=True)
        
        # 입력 방법 선택
        col1, col2 = st.columns([4,1])
        prompt = ""
        
        with col1:
            # 채팅 입력 위젯은 항상 여기에 있어야 하며, 세션 상태에 저장
            prompt = st.chat_input("일정 입력 또는 브리핑을 요청해보세요.", key="chat_input_key")
        
        with col2:
            # 음성 입력 버튼
            if st.button("🎤 음성 입력", key="voice_input_button"):
                st.session_state.user_prompt = 음성입력()
                st.rerun() # 음성 입력 후 프롬프트를 처리하기 위해 새로고침

        # prompt가 입력되면 세션 상태에 저장하고 새로고침
        if prompt:
            st.session_state.user_prompt = prompt
            st.rerun() # main() 함수에서 프롬프트를 처리하도록 새로고침
            
        # 하단 고정 컨테이너 닫기
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    # 채팅 메시지 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # **[핵심 변경]** fixed_input_bar()에서 설정된 prompt를 여기서 가져와 처리
    prompt = st.session_state.user_prompt
    st.session_state.user_prompt = None # 처리 후 초기화

    if prompt:
        # 사용자 메시지 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 서버 요청 (기존 로직 유지)
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
        
        # 상세 기록 확장 (기존 로직 유지)
        with st.expander("Detailed Processes"):
            if st.session_state.last_history:
                for idx, step in enumerate(st.session_state.last_history, 1):
                    with st.expander(f"Step {idx} — {step['node']}"):
                        st.json(step["state"])
            else:
                st.info("아직 실행 기록이 없습니다. 왼쪽에서 요청을 실행하세요.")


def sidebar_main():
    st.subheader("🔐 Google OAuth 인증 관리")
    
    secret_file = st.file_uploader("Upload secret.json", type=["json"])
    if secret_file is not None:
        st.success(f"파일 선택됨: {secret_file.name}")
        
        if st.button("Send to backend"):
            try:
                # 파일 데이터 준비
                files = {
                    "secret_file": (secret_file.name, secret_file, "application/json")
                }
                
                # 요청 전송
                response = requests.post(
                    "http://localhost:8000/schedule/upload_keys", 
                    files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ 인증 성공!")
                    st.json(result)
                else:
                    st.error(f"❌ 오류 발생: {response.status_code}")
                    st.text(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚫 백엔드 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")

    with st.expander("📖 사용 방법"):
        st.markdown("""
        1. **클라이언트 시크릿 파일 업로드**
           - Google Cloud Console에서 OAuth 2.0 클라이언트 ID를 생성
           - JSON 형식의 클라이언트 시크릿 파일 다운로드
           - 위에서 파일 업로드 (백엔드 저장)
        
        2. **OAuth 인증 수행 (백엔드에서)**
           - 'client_secret'이 포함된 파일이 확인되면 '토큰 생성 시작' 버튼 활성화
           - 버튼 클릭 후 브라우저에서 OAuth 인증 진행
           - Google 계정 로그인 및 권한 승인
        
        3. **토큰 확인(벡엔드에서)**
           - 인증 성공 시 `token.json` 파일 생성
           - 이후 API 호출에 사용 가능
        
        **주의사항:**
        - 업로드된 파일은 백앤드에 `keys/secret.json`으로 저장됨
        - 토큰 파일은 백엔드에 `keys/token.json`으로 저장됨
        - 민감한 정보가 포함되어 있으므로 안전하게 보관하세요
        """)


if __name__ == "__main__":

    with st.sidebar:
        sidebar_main()

    st.title("Schedule Agent")
    st.markdown("---")

    import streamlit as st

    main()
    대화초기화() # 함수 호출 위치를 main() 바깥, 제목 아래로 옮겼습니다.
    
    # **[핵심 변경]** 가장 마지막에 하단 입력창을 렌더링
    fixed_input_bar()


# import streamlit as st
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# import pandas as pd
# import datetime
# import plotly.express as px

# # -------------------------------
# # OAuth 인증
# # -------------------------------
# SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# st.title("Google Calendar Viewer")

# if "creds" not in st.session_state:
#     st.session_state.creds = None

# if st.session_state.creds is None:
#     st.write("**구글 계정으로 로그인 필요**")
#     if st.button("Login with Google"):
#         flow = InstalledAppFlow.from_client_secrets_file(
#             'D:/Agents/backend/config/client_secret_39562377782-nge5sdugil9eurkbgn54temjtgq06tbh.apps.googleusercontent.com.json', SCOPES
#         )
#         creds = flow.run_local_server(port=0)
#         st.session_state.creds = creds
#         st.rerun()  # 로그인 후 새로고침

# # -------------------------------
# # Google Calendar API 호출
# # -------------------------------
# if st.session_state.creds:
#     service = build('calendar', 'v3', credentials=st.session_state.creds)
#     now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
#     events_result = service.events().list(
#         calendarId='primary', timeMin=now, maxResults=50, singleEvents=True,
#         orderBy='startTime'
#     ).execute()
#     events = events_result.get('items', [])

#     if not events:
#         st.write("캘린더에 예정된 이벤트가 없습니다.")
#     else:
#         # -------------------------------
#         # 이벤트 데이터를 DataFrame으로 변환
#         # -------------------------------
#         data = []
#         for event in events:
#             start = event['start'].get('dateTime', event['start'].get('date'))
#             end = event['end'].get('dateTime', event['end'].get('date'))
#             data.append({'start': start, 'end': end, 'summary': event.get('summary', 'No Title')})
        
#         df = pd.DataFrame(data)
#         df['start'] = pd.to_datetime(df['start'])
#         df['end'] = pd.to_datetime(df['end'])

#         st.write("### Upcoming Events")
#         st.dataframe(df[['start', 'end', 'summary']])

#         # -------------------------------
#         # 달력 형태 시각화 (막대 차트)
#         # -------------------------------
#         fig = px.timeline(df, x_start="start", x_end="end", y="summary", color="summary")
#         fig.update_yaxes(autorange="reversed")  # 상단부터 최근 이벤트
#         st.write("### Calendar View")
#         st.plotly_chart(fig, use_container_width=True)