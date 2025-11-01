import streamlit as st
from streamlit.components.v1 import html
import requests
import uuid
import json

st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

# Configuration
BACKEND_URL = "http://localhost:8000"  # Update with your FastAPI server URL


canvas_html = """
<canvas id="geometryCanvas"></canvas>
"""

# Initialize session state
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = {} # 각 질문에 대한 검색 결과를 저장하도록 딕셔너리로 변경
    if "thinking_process" not in st.session_state:
        st.session_state.thinking_process = {} # 각 메시지의 thinking 과정을 저장
    
initialize_session_state()

def sidebar():
    with st.sidebar:
        st.header("Settings")  
        doc_top_k = st.slider("Document Top K", 1, 20, 10)
        rerank_k = st.slider("Rerank K", 1, 10, 5)
        rerank_threshold = st.slider("Rerank Threshold", 0.0, 1.0, 0.0)
        search_type = st.pills("Search Type", options=["hybrid_search"], default=["hybrid_search"], selection_mode="single")
        
        # New conversation button
        if st.button("🔄 New Conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.conversation_history = []
            st.session_state.search_results = {} # 새 대화 시작 시 초기화
            st.session_state.thinking_process = {} # 새 대화 시작 시 초기화
            st.rerun()
        st.session_state.session_id

        return doc_top_k,  rerank_k, rerank_threshold, search_type


def chat_history():
    # Display conversation history
    for i, message in enumerate(st.session_state.conversation_history):
        with st.chat_message(message["role"]):
            # Thinking process display
            if message["role"] == "assistant" and st.session_state.thinking_process.get(i):
                with st.expander("💭 Thinking..."):
                    st.markdown(st.session_state.thinking_process[i])
            
            st.markdown(message["content"])
            
            # Sources display
            if message["role"] == "assistant" and st.session_state.search_results.get(i):
                with st.expander("📖 참고문서"):
                    sources_to_display = []
                    for doc in st.session_state.search_results[i]:
                        source_content = doc.get("page_content", "내용 없음")
                        source_metadata = doc.get("metadata", "내용 없음")
                        with st.container(height=300):
                            st.warning(source_metadata)
                            st.info(source_content)

def main(doc_top_k:int,rerank_k:int, rerank_threshold:float, search_type:str):
    # User input
    if prompt := st.chat_input("Ask me anything"):

        # 멀티턴 질문 정제
        query_payload = {"question": prompt, "chat_history": st.session_state.conversation_history}
        refined_result = requests.post(f"{BACKEND_URL}/refine", json=query_payload)
        refined_result = refined_result.json()


        # Append user message immediately
        st.session_state.conversation_history.append({
            "role": "user",
            "content": refined_result["refined_query"]
        })
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare for assistant's response
        # 현재 답변이 들어갈 인덱스를 미리 저장 (스트리밍 시 업데이트를 위함)
        assistant_message_index = len(st.session_state.conversation_history) 
        st.session_state.conversation_history.append({"role": "assistant", "content": "..."}) # Placeholder for streaming
        
        # Placeholders for streaming content (outside the loop for proper update)
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            answer_placeholder = st.empty()
            sources_placeholder = st.empty() # 소스를 위한 플레이스홀더 추가
    

        # Step 1: Perform search
        search_payload = {
            "question": refined_result["refined_query"],
            "top_k": {"doc": doc_top_k},
            "rerank_k": rerank_k,
            "rerank_threshold": rerank_threshold,
            "session_id": st.session_state["session_id"]
        }

        try:
            print("---- Start Search ----")
            search_response = requests.post(f"{BACKEND_URL}/{search_type}", json=search_payload)
            search_response.raise_for_status()
            search_data = search_response.json()
            print("---- Search Result ----")            
            # Store search results for this specific assistant response
            current_search_results = search_data["documents"]
            st.session_state.search_results[int(assistant_message_index)] = current_search_results
            
            # Step 2: Generate answer with streaming
            print("---- Start Generation ----")
            full_response_content = ""
            thinking_content = ""
            
            # Prepare generation request
            gen_payload = {
                "question": refined_result["refined_query"], #search_data["refined_question"],
                "questions": [msg["content"] for msg in st.session_state.conversation_history if msg["role"] == "user"],
                "documents": current_search_results,
                "session_id": st.session_state.session_id
            }
            
            # Stream the response
            with requests.post(f"{BACKEND_URL}/generate", json=gen_payload, stream=True) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            try:
                                data = json.loads(decoded_line[5:].strip())
                                
                                if data.get("type") in ["thinking_start", "thinking", "thinking_end"]:
                                    if data.get("type") == "thinking":
                                        thinking_content += data.get("content", "")
                                    
                                    # Update thinking placeholder with expander
                                    with thinking_placeholder.container():
                                        with st.expander("AI is thinking...", expanded=True):
                                            st.markdown(thinking_content)
                                    
                                elif data.get("type") == "answer":
                                    full_response_content += data.get("content", "")
                                    answer_placeholder.markdown(full_response_content + "▌") # Add cursor
                            
                            except json.JSONDecodeError:
                                continue
                
                # After streaming, update final answer and thinking content
                answer_placeholder.markdown(full_response_content) # Remove cursor
                
                # Store final thinking content in session state
                if thinking_content:
                    st.session_state.thinking_process[assistant_message_index] = thinking_content
                    with thinking_placeholder.container():
                        with st.expander("Thought Process", expanded=False): # Collapse after response
                            st.markdown(thinking_content)
                else:
                    thinking_placeholder.empty() # Clear if no thinking content
                
                # Update the stored conversation history with the full content
                st.session_state.conversation_history[assistant_message_index]["content"] = full_response_content

                # Display sources in its own expander after the answer
                if current_search_results:
                    with sources_placeholder.container():
                        with st.expander("참고문서"):
                            sources_to_display = []
                            for doc in current_search_results:
                                source_content = doc.get("page_content", "내용 없음")
                                source_metadata = doc.get("metadata", "내용 없음")
                                with st.container(height=300):
                                    st.warning(source_metadata)
                                    st.info(source_content)
                
                st.rerun() # Rerun to properly display updated history and clear input box
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {str(e)}")
            # Remove the placeholder message if an error occurs
            if len(st.session_state.conversation_history) > assistant_message_index:
                st.session_state.conversation_history[assistant_message_index]["content"] = f"Error: {str(e)}"
            st.rerun()

        debug()

def debug():
    # Debug section (optional)
    with st.expander("Debug Info"):
        st.write("Session ID:", st.session_state.session_id)
        st.write("Current Search Results (by index):", st.session_state.search_results)
        st.write("Thinking Processes (by index):", st.session_state.thinking_process)
        st.write("Conversation History:", st.session_state.conversation_history)

if __name__ == "__main__":
    st.title("Rag Agent")
    doc_top_k, rerank_k, rerank_threshold, search_type = sidebar()
    chat_history()
    main(doc_top_k, rerank_k, rerank_threshold, search_type)
    
