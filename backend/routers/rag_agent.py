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
MODEL_NAME = os.getenv("BIG_MODEL")
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