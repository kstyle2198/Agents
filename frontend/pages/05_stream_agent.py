from dotenv import load_dotenv
load_dotenv()
import uuid
import streamlit as st
import requests

st.title("LangGraph Streaming Test")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "ai", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    role = "assistant" if msg['role'] == "ai" else "user"
    st.chat_message(role).write(msg["content"])

def stream_response(messages: list):
    """Generator function to stream response from backend"""
    url = "http://localhost:8000/invoke"  # Update to your streaming endpoint
    try:
        with requests.post(url, json={"messages": messages}, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to server: {str(e)}"

if prompt := st.chat_input():
    # Add user message to history and display
    st.session_state.messages.append({"role": "human", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare assistant message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in stream_response(st.session_state.messages):
            full_response += chunk
            response_placeholder.markdown(full_response + " ")  # Add typing indicator
            import time
            time.sleep(0.5)
        # Finalize the response display and update state
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "ai", "content": full_response})

if st.button("🔄 New Conversation"):
    st.session_state.messages = []
    st.rerun()
