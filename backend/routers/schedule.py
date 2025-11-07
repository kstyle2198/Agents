import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(override=True)

# 로거 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

llm = ChatGroq(temperature=0, model_name= "llama-3.3-70b-versatile") 

from utils.schedule_helper import (
    add_meeting_event,
    event_dict,
    delete_event,
    get_event_id,
    is_new_event,
    process_schedule_request,
    get_schedules,
    schedule_briefing
)


from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import asyncio

# ==== FastAPI 초기화 ====
from fastapi import APIRouter, HTTPException
schedule = APIRouter(prefix="/schedule")

# ==== LangGraph 상태 모델 ====
from typing import Any, List, Dict
class ScheduleState(BaseModel):
    user_input: str = ""
    route: Optional[Literal["command", "question"]] = None
    my_schedule: Optional[List[Dict[str, Any]]] = None  # ✅ 리스트로 변경
    calendar_id: str = ""
    job_type: Optional[str] = None
    old_time: Optional[datetime] = None
    old_info: Optional[str] = None
    new_time: Optional[datetime] = None
    new_info: Optional[str] = None
    del_time: Optional[datetime] = None
    del_info: Optional[str] = None
    events: Optional[dict] = None
    event_list: Optional[list] = None
    event_id: Optional[str] = None
    is_new: Optional[bool] = None
    result: Optional[str] = None

# ==== LangGraph 노드 ====

### Routing
async def router_node(state: ScheduleState) -> ScheduleState:
    """입력을 지시형 / 질문형으로 라우팅"""
    text = str(state.user_input)
    prompt = f"""
    아래 문장이 '지시형'(추가, 삭제, 변경 등)인지 '질문형'(브리핑, 설명, why/how/무엇 등)인지 판단하세요.
    문장: "{text}"
    답변은 command 또는 question 중 하나로만 하세요.
    """
    route = llm.invoke(prompt).content.strip().lower()
    if route not in ["command", "question"]:
        route = "question"  # fallback
    logger.info(f"route_node: {route}")
    state.route = route
    return state

def conditional_router(state: ScheduleState):
    return state.route

### Secretary
async def get_schedule_node(state: ScheduleState) -> ScheduleState:
    my_schedule = get_schedules(user_input=state.user_input, calendar_id=state.calendar_id)
    state.my_schedule = my_schedule
    logger.info(f"my_schedule: {my_schedule}")
    return state

async def brief_schedule_node(state: ScheduleState) -> ScheduleState:
    briefing = schedule_briefing(str(state.user_input), event_dict=str(state.my_schedule))
    state.result = briefing
    logger.info(f"briefing: {briefing}")
    return state

### Scheduler
async def parse_request_node(state: ScheduleState) -> ScheduleState:
    parsed = process_schedule_request(state.user_input)
    if parsed:
        s = parsed[0]
        state.job_type = s.job_type
        state.old_time = s.old_time
        state.old_info = s.old_info
        state.new_time = s.new_time
        state.new_info = s.new_info
        state.del_time = s.del_time
        state.del_info = s.del_info
    logger.info(f"parse_request_node: {state}")
    return state

async def fetch_events_node(state: ScheduleState) -> ScheduleState:
    if state.job_type == "New":
        date_only = datetime(state.new_time.year, state.new_time.month, state.new_time.day)
    elif state.job_type == "Update":
        date_only = datetime(state.old_time.year, state.old_time.month, state.old_time.day)
    elif state.job_type == "Delete":
        date_only = datetime(state.del_time.year, state.del_time.month, state.del_time.day)
    else:
        return state
    state.events = event_dict(state.calendar_id, date_only)
    state.event_list = list(state.events.keys())
    logger.info(f">>> Fetched Event List: {state.event_list}")
    return state

async def check_new_event_node(state: ScheduleState) -> ScheduleState:
    if state.job_type == "New":
        state.is_new = is_new_event(state.user_input, state.event_list)
    elif state.job_type == "Update":
        state.is_new = is_new_event(state.old_info, state.event_list)
    logger.info(f">>> Check New : {state.is_new}")
    return state

async def get_event_id_node(state: ScheduleState) -> ScheduleState:
    if state.events:
        try:
            event_id_data = get_event_id(state.user_input, state.events)
            state.event_id = event_id_data[0]["events"][0]["id"]
            logger.info(f">>> Event ID : {state.event_id}")
        except Exception as e:
            state.event_id = None
            logger.warning(f">>> Event ID : {state.event_id}")
    return state

async def process_job_node(state: ScheduleState) -> ScheduleState:
    logger.info(f">>> State : {state}")
    if state.job_type == "New" and state.is_new:
        add_meeting_event(state.calendar_id, state.new_time, state.new_info)
        state.result = f"✅ 일정 추가 완료: {state.new_time} - {state.new_info}"
        logger.info(f">>> State Info : {state.result}")

    elif state.job_type == "Update" and state.is_new is False:
        delete_event(state.calendar_id, state.event_id)
        add_meeting_event(state.calendar_id, state.new_time, state.new_info)
        state.result = f"♻ 일정 수정 완료: {state.new_time} - {state.new_info}"
        logger.info(f">>> State Info : {state.result}")

    elif state.job_type == "Delete":
        delete_event(state.calendar_id, state.event_id)
        state.result = f"🗑 일정 삭제 완료: {state.del_time} - {state.del_info}"
        logger.info(f">>> State Info : {state.result}")

    else:
        state.result = "⚠ 처리할 일정이 없습니다."
        logger.warning(f">>> State Info : {state.result}")
    return state

# ==== LangGraph 구성 ====
from langgraph.graph import StateGraph, START, END

def secretary_builder(state):
    graph = StateGraph(state)
    graph.add_node("get_schedule", get_schedule_node)
    graph.add_node("brief_schedule", brief_schedule_node)

    graph.add_edge(START, "get_schedule")
    graph.add_edge("get_schedule", "brief_schedule")
    graph.add_edge("brief_schedule", END)
    return graph.compile()

def scheduler_builder(state):
    graph = StateGraph(state)

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("fetch_events", fetch_events_node)
    graph.add_node("check_new", check_new_event_node)
    graph.add_node("get_event_id", get_event_id_node)
    graph.add_node("process_job", process_job_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "fetch_events")
    graph.add_edge("fetch_events", "check_new")
    graph.add_edge("check_new", "get_event_id")
    graph.add_edge("get_event_id", "process_job")
    graph.add_edge("process_job", END)
    return graph.compile()

def main_builder(state):
    workflow = StateGraph(state)
    workflow.add_node("router", router_node)
    workflow.add_node("secretary", secretary_builder(state=state))
    workflow.add_node("scheduler", scheduler_builder(state=state))

    # Build graph
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        conditional_router,
        {
            "command": "scheduler",
            "question": "secretary",
            "__end__": END
        },
    )
    workflow.add_edge("scheduler", END)
    workflow.add_edge("secretary", END)

    return workflow.compile()


schedule_graph = main_builder(state=ScheduleState)

# ==== FastAPI Endpoint ====
class ScheduleRequest(BaseModel):
    user_input: str
    calendar_id: str

@schedule.post("/", tags=["Scheduler"])
async def schedule_handler(req: ScheduleRequest):
    initial_state = ScheduleState(user_input=req.user_input, calendar_id=req.calendar_id)
    history = []

    async for step_data in schedule_graph.astream(initial_state):
        for node_name, node_state in step_data.items():
            history.append({
                "node": node_name,
                "state": node_state.dict() if hasattr(node_state, "dict") else dict(node_state)
            })

    final_state = history[-1]["state"] if history else {}
    return {
        "result": final_state.get("result", "⚠ 결과 없음"),
        "history": history
    }

if __name__ == "__main__":
    pass