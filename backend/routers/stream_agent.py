from typing import Annotated, TypedDict, List

from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_groq import ChatGroq

import os
NO_THINK_MODEL = os.getenv("NO_THINK_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define FastAPI Router
from fastapi import APIRouter, HTTPException
stream_agent = APIRouter()

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list, add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Core invocation of the model
# def _call_model(state: GraphsState):
#     messages = state["messages"]
#     llm = ChatGroq(model_name=NO_THINK_MODEL, temperature=0.0, streaming=True)
#     response = llm.invoke(messages)
#     return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm


# Use Groq endpoint instead of OpenAI
from openai import OpenAI
from groq import Groq

def _call_model(state:GraphsState):
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    # 시스템 프롬프트 정의
    system_prompt = """You are a helpful AI assistant. Please provide clear, accurate, and helpful responses to the user's questions within 10 lines.""" 
    # Be concise yet thorough in your explanations, and maintain a professional and friendly tone throughout the conversation."""

    # 전체 메시지 히스토리를 API에 전달
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in state["messages"]:
        # 메시지 형식을 OpenAI API 형식으로 변환
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            api_messages.append({"role": msg.role, "content": msg.content})
        else:
            # 다른 형식의 메시지 객체 처리
            api_messages.append({"role": "user", "content": str(msg)})
    try:
        response = client.chat.completions.create(
            model=NO_THINK_MODEL,
            messages=api_messages,
        )
        assistant_message = response.choices[0].message.content        
        # 올바른 형식으로 메시지 추가
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": assistant_message
                }
            ]
        }
        
    except Exception as e:
        print(f"API call failed: {e}")
        # 에러 처리
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": "Sorry, I encountered an error processing your request."
                }
            ]
        }

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", END)

# Compile the state graph into a runnable object
graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})

# Define Pydantic model for request body
from typing import List, Any, Dict
class RequestModel(BaseModel):
    messages: List


@stream_agent.post("/invoke", tags=["Stream_Agent"])
def invoke_graph(request: RequestModel):
    try:
        result = invoke_our_graph(request.messages, [])
        return result["messages"][-1].content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))