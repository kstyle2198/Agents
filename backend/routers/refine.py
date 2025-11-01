from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, Dict, Optional, Union, TypedDict, Annotated

# 로거 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

import os
from dotenv import load_dotenv
load_dotenv(override=True)
think_model = os.getenv("THINK_MODEL")
no_think_model = os.getenv("NO_THINK_MODEL")

llm = ChatGroq(model_name=think_model, temperature=0, max_tokens=3000,) 

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class RefineState(TypedDict):
    """State for query refinement"""
    question: str
    chat_history: list
    refined_query: str  

def format_query(docs: list) -> str:
    """Format documents for context"""
    try:
        return "\n\n".join(doc["content"].strip().replace("\n", "") for doc in docs[-8:])
    except:
        return ""
    
import re
def remove_think_tags(text):
    """
    <think> ~ </think> 태그와 그 내용을 제거합니다.
    """
    logger.debug(f"remove_think_tags")
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)

from langchain_core.messages import HumanMessage
def refined_question(state:RefineState):
    """Refine the question based on chat history"""
    question = state['question']
    chat_history = state['chat_history']
    format_chat_history = format_query(chat_history)

    try:
        prompt = f"""아래 질문 내용이 chat_history와 관련이 있는 경우, chat_history를 참고하여 질문을 정제해 주세요.
        chat_history의 내용이 없거나, 내용이 질문과 상관이 없는 경우, 질문 자체의 표현만 정제해주세요.
        질문에 대해 답하지 말고, 표현만 정제해주세요.
        <question>
        {question}

        <chat_history>
        {format_chat_history}
        """
        refined_question_content = llm.invoke([HumanMessage(content=prompt)]).content
        refined_question_content = remove_think_tags(refined_question_content)
        return {"refined_query": refined_question_content}
    except Exception as e:
        logger.exception("Error occurred during Refine")
        raise

def query_refiner(state):
    """Build the refinement graph"""
    refine_builder = StateGraph(state)
    refine_builder.add_node("refine_agent", refined_question)

    refine_builder.add_edge(START, "refine_agent")
    refine_builder.add_edge("refine_agent", END) 
    return refine_builder.compile()

refine_graph = query_refiner(RefineState)

class RefineRequest(BaseModel):
    """Request model for query refinement"""
    question: str
    chat_history: list = []

class RefineResponse(BaseModel):
    """Response model for query refinement"""
    refined_query: str


from fastapi import APIRouter, HTTPException
refine = APIRouter()

@refine.post("/refine", response_model=RefineResponse, tags=["Refine"], operation_id="refine_question")
async def refine_question(req: RefineRequest):
    logger.info(f"Refine the query: {req.question}")
    try:
        # 초기 상태 구성
        state = {
            "question": req.question,
            "chat_history": req.chat_history
            }

        # 그래프 실행
        result = refine_graph.invoke(state)
        logger.info(f"Refining Query completed. - {result["refined_query"]}")
        return RefineResponse(refined_query=result["refined_query"])

    except Exception as e:
        logger.error("Query Refinery failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
