# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from typing import TypedDict, List, Any, Dict

# from langchain_groq import ChatGroq
# from elasticsearch import Elasticsearch

# from process.es_search_index import es
# from langgraph.checkpoint.memory import MemorySaver

# from utils.setlogger import setup_logger
# logger = setup_logger(f"{__name__}")

# import os
# from dotenv import load_dotenv
# load_dotenv(override=True)
# NO_THINK_MODEL = os.getenv("NO_THINK_MODEL")

# # ---------------------------------------------------------
# # 상태 정의 (멀티턴)
# # ---------------------------------------------------------
# class QAState(TypedDict):
#     messages: List[BaseMessage]        # 대화 로그
#     refined_query: str                 # 정제된 쿼리
#     search_results: List[Dict[str, Any]]  # ES 검색 결과
#     answer: str                        # 최종 응답


# # ---------------------------------------------------------
# # LLM 설정
# # ---------------------------------------------------------
# llm_refiner = NO_THINK_MODEL
# llm_answer = NO_THINK_MODEL

# # ---------------------------------------------------------
# # Node 1: Query Refiner
# # ---------------------------------------------------------
# def refine_query(state: QAState):
#     user_msg = state["messages"][-1].content

#     prompt = f"""
# 다음 사용자의 질문을 검색 친화적인 쿼리로 다시 정제해줘.
# 과도하게 길 필요는 없고 핵심 검색어 중심으로 재작성.

# 사용자 질문:
# {user_msg}
# """

#     refined = llm_refiner.invoke(prompt).content
#     state["refined_query"] = refined
#     return state


# # ---------------------------------------------------------
# # Node 2: Elasticsearch Search
# # ---------------------------------------------------------
# def es_search(state: QAState):
#     query = state["refined_query"]
#     docs = es.search_documents(query_text=query)
#     state["search_results"] = docs
#     return state

# # ---------------------------------------------------------
# # Node 3: Answer Generator (LLM)
# # ---------------------------------------------------------
# def answer_question(state: QAState):
#     user_msg = state["messages"][-1].content
#     refined = state["refined_query"]
#     docs = state["search_results"]

#     docs_str = "\n\n".join([f"- {d['content']}" for d in docs])

#     prompt = f"""
# 당신은 Elasticsearch 기반 QA 시스템입니다.

# [정제된 질문]
# {refined}

# [검색 결과]
# {docs_str}

# 이 정보를 기반으로 사용자의 질문에 정확하고 간결하게 답변하세요.

# 사용자 질문: {user_msg}
# """

#     answer = llm_answer.invoke(prompt).content
#     state["messages"].append(AIMessage(content=answer))
#     state["answer"] = answer
#     return state


# # ---------------------------------------------------------
# # 그래프 구성
# # ---------------------------------------------------------
# graph = StateGraph(QAState)

# graph.add_node("refine_query", refine_query)
# graph.add_node("search", es_search)
# graph.add_node("answer", answer_question)

# graph.add_edge(START, "refine_query")
# graph.add_edge("refine_query", "search")
# graph.add_edge("search", "answer")
# graph.add_edge("answer", END)

# memory_saver = MemorySaver()
# app = graph.compile(checkpointer=memory_saver)



# from uuid import uuid4
# from typing import Optional
# from pydantic import BaseModel, Field
# from fastapi import APIRouter, HTTPException
# from langgraph.errors import GraphRecursionError
# from fastapi.responses import StreamingResponse
# from langchain_core.messages import HumanMessage, AIMessageChunk

# rag_agent = APIRouter()


# class ChatRequest(BaseModel):
#     question: str
#     session_id: Optional[str] = None  # 없으면 새 세션 생성


# # ---------------------------------------------------------
# # Response Schema
# # ---------------------------------------------------------
# class ChatResponse(BaseModel):
#     answer: str
#     session_id: str


# # ---------------------------------------------------------
# # Endpoint
# # ---------------------------------------------------------
# @rag_agent.post("/rag_chat", response_model=ChatResponse, tags=["rag_agent"])
# async def chat(req: ChatRequest):

#     # 1) 세션 유지: MemorySaver가 session_id 기반으로 대화 연결
#     config = {"configurable": {"thread_id": req.session_id or "default"}}

#     # 2) LangGraph 호출 - HumanMessage로 입력
#     result = app.invoke(
#         input={"messages": [HumanMessage(content=req.question)]},
#         config=config
#     )

#     # 3) 결과에서 answer 추출
#     answer = result["answer"]

#     # MemorySaver는 thread_id(=session_id)를 자동 생성 가능
#     session_id = config["configurable"]["thread_id"]

#     return ChatResponse(
#         answer=answer,
#         session_id=session_id
#     )


# @rag_agent.get("/rag_threads", tags=["rag_agent"])
# def list_threads(thread_id: str):
#     try:
#         # MemorySaver에 저장된 모든 스레드 목록 조회
#         threads = list(app.get_state_history(config={"configurable": {"thread_id": thread_id}}))
#         logger.info(f"Retrieved history for thread_id {thread_id}: {threads}")
#         # 임시 응답 (실제 구현에 맞게 수정 필요)
#         return {"threads": threads}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


import os
import uuid
import json
import asyncio
from typing import TypedDict, List, Any, Dict, Annotated, AsyncGenerator

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# Custom Imports
from process.es_search_index import es
from utils.setlogger import setup_logger

load_dotenv(override=True)
logger = setup_logger(f"{__name__}")

# ---------------------------------------------------------
# 1. LLM 및 State 설정 (이전과 동일)
# ---------------------------------------------------------
MODEL_NAME = os.getenv("NO_THINK_MODEL", "llama3-8b-8192")
llm = ChatGroq(model=MODEL_NAME, temperature=0)

class QAState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    refined_query: str
    search_results: List[Dict[str, Any]]
    answer: str

# ---------------------------------------------------------
# 2. Node 정의 (로직 동일)
# ---------------------------------------------------------
async def refine_query(state: QAState):
    messages = state["messages"]
    last_message = messages[-1].content
    
    if len(messages) <= 1:
        prompt = f"다음 질문을 Elasticsearch 검색에 적합한 키워드 중심으로 간결하게 변환해줘:\n{last_message}"
        refined = await llm.ainvoke(prompt) # Async 호출 권장
        refined_text = refined.content
    else:
        # 멀티턴 처리 로직
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question..."
            "(중략)... just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history = messages[:-1]
        chain = contextualize_q_prompt | llm
        refined = await chain.ainvoke({"chat_history": history, "input": last_message})
        refined_text = refined.content

    return {"refined_query": refined_text}

async def es_search(state: QAState):
    query = state["refined_query"]
    # ES는 동기 함수일 수 있으므로, 필요하다면 run_in_executor 등으로 감쌀 수 있음
    # 여기서는 간단히 호출
    try:
        docs = es.search_documents(query_text=query)
    except Exception:
        docs = []
    return {"search_results": docs}

async def answer_question(state: QAState):
    docs = state["search_results"]
    messages = state["messages"]
    
    context_text = "\n\n".join([f"- {d.get('content', '')}" for d in docs]) if docs else "No results."
    
    qa_system_prompt = f"""
    당신은 Elasticsearch 기반 AI 어시스턴트입니다.
    [Context]를 바탕으로 답변하세요.
    
    [Context]
    {context_text}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
        ]
    )
    
    chain = qa_prompt | llm
    
    # Streaming을 위해 여기서는 단순히 리턴하지 않고 실행 흐름만 정의
    # 실제 값은 그래프 실행 시 stream events로 캡처됨
    response = await chain.ainvoke({"chat_history": messages})
    
    return {"answer": response.content, "messages": [response]}

# ---------------------------------------------------------
# 3. 그래프 구성
# ---------------------------------------------------------
workflow = StateGraph(QAState)
workflow.add_node("refine_query", refine_query)
workflow.add_node("search", es_search)
workflow.add_node("answer", answer_question)

workflow.add_edge(START, "refine_query")
workflow.add_edge("refine_query", "search")
workflow.add_edge("search", "answer")
workflow.add_edge("answer", END)

memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)

# ---------------------------------------------------------
# 4. FastAPI Endpoint (Streaming 구현)
# ---------------------------------------------------------
rag_agent = APIRouter()

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

# 스트리밍 제너레이터 함수
async def generate_chat_stream(input_data: dict, config: dict, session_id: str) -> AsyncGenerator[str, None]:
    """
    LangGraph의 이벤트를 감지하여 클라이언트에게 실시간 전송합니다.
    형식: JSON String (line-delimited)
    """
    
    # astream_events: 그래프 내부의 모든 이벤트를 스트리밍 (v2 버전 사용 권장)
    async for event in rag_app.astream_events(input_data, config=config, version="v2"):
        
        kind = event["event"]
        
        # 1. 'answer' 노드 내부의 LLM 스트리밍 이벤트만 필터링
        if kind == "on_chat_model_stream":
            # 메타데이터를 확인하여 현재 실행 중인 노드가 'answer'인지 확인
            # (refine_query 노드의 LLM 생성 결과는 사용자에게 보여주지 않음)
            if event["metadata"].get("langgraph_node") == "answer":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    # 토큰 전송
                    yield json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False) + "\n"
    
    # 2. 스트림 종료 후 세션 ID 전송 (클라이언트가 세션을 저장할 수 있게 함)
    yield json.dumps({"type": "end", "session_id": session_id}, ensure_ascii=False) + "\n"


@rag_agent.post("/rag_chat_stream", tags=["rag_agent"])
async def chat_stream(req: ChatRequest):
    # 세션 ID 생성 또는 유지
    session_id = req.session_id if req.session_id else str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    input_data = {"messages": [HumanMessage(content=req.question)]}
    
    return StreamingResponse(
        generate_chat_stream(input_data, config, session_id),
        media_type="application/x-ndjson" # Newline Delimited JSON
    )

@rag_agent.get("/rag_threads", tags=["rag_agent"])
def list_threads(thread_id: str):
    try:
        # MemorySaver에 저장된 모든 스레드 목록 조회
        threads = list(rag_app.get_state_history(config={"configurable": {"thread_id": thread_id}}))
        logger.info(f"Retrieved history for thread_id {thread_id}: {threads}")
        # 임시 응답 (실제 구현에 맞게 수정 필요)
        return {"threads": threads}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))