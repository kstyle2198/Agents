import streamlit as st
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv(override=True)

# --- 캐싱을 사용한 비동기 함수 설정 ---

@st.cache_resource(show_spinner="Connecting to Tool-Server...")
def get_tools():
    """
    MCP 서버에 연결하여 사용 가능한 도구를 가져옵니다.
    Streamlit의 @st.cache_resource를 사용하여 앱 세션 동안 도구 목록을 한 번만 로드합니다.
    """
    server_config = {
        "search": {
            "url": "http://localhost:8000/mcp",
            "transport": "sse",
        },
    }
    
    async def fetch_tools():
        client = MultiServerMCPClient(server_config)
        return await client.get_tools()

    # Streamlit의 동기적 컨텍스트에서 비동기 함수 실행
    try:
        return asyncio.run(fetch_tools())
    except Exception as e:
        st.error(f"Failed to connect to the tool server: {e}")
        return []

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")
st.title("MCP Agent")
st.caption("A Streamlit app for LangGraph ReAct agent with MCP Tools")

# 사용 가능한 도구 로드 및 표시
tools = get_tools()
if tools:
    tool_names = [tool.name for tool in tools]
    st.sidebar.success("✅ Tool-Server Connected")
    st.sidebar.write("Available Tools:")
    for name in tool_names:
        st.sidebar.code(name, language="text")
else:
    st.sidebar.error("❌ Tool-Server Disconnected")
    st.warning("Tool server is not available. The agent might not function correctly.")

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 에이전트 및 모델 설정 ---

# LLM 정의
model = ChatGroq(
    model="qwen/qwen3-32b", # 사용 가능한 모델로 변경 가능
    temperature=0.5,
    max_tokens=2000,
)

# 프롬프트 정의
prompt_template = """
You are the Smart AI Assistant in a company.
Based on the result of tool calling, Generate a consice and logical answer.
and if there is no relevant infomation in the tool calling result, Just say 'I don't know'.
Answer in Korean.
"""

# ReAct 에이전트 생성
# tools가 비어있지 않은 경우에만 에이전트를 생성합니다.
if tools:
    agent = create_react_agent(model=model, tools=tools, prompt=prompt_template)

# 사용자 입력 처리
if user_query := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 어시스턴트 응답 생성 및 표시
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # LangGraph 에이전트 스트림 호출
            inputs = {"messages": [("user", user_query)]}
            
            # 스트리밍 응답을 비동기적으로 처리하고 최종 결과를 반환하는 함수
            async def stream_and_get_response():
                # 이 함수 내의 지역 변수로 full_response를 사용합니다.
                _full_response = "" 
                
                # ainvoke는 비동기 제너레이터를 반환합니다.
                async for chunk in agent.astream(inputs, stream_mode="values"):
                    # 스트림의 마지막 메시지(AIMessage)의 내용을 가져옵니다.
                    if "messages" in chunk and isinstance(chunk["messages"][-1], AIMessage):
                        message_content = chunk["messages"][-1].content
                        # 이전 내용과 다를 경우에만 업데이트하여 깜빡임 방지
                        if message_content != _full_response:
                            _full_response = message_content
                            response_placeholder.markdown(_full_response + "▌") # 커서 효과 추가
                
                response_placeholder.markdown(_full_response) # 최종 응답 표시
                return _full_response # 최종 결과를 반환합니다.

            # Streamlit에서 비동기 스트림을 실행하고, 반환된 값을 full_response에 저장합니다.
            full_response = asyncio.run(stream_and_get_response())

        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "죄송합니다. 답변을 생성하는 동안 오류가 발생했습니다."
            response_placeholder.markdown(full_response)

    # 최종 응답을 채팅 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": full_response})