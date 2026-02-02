import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import Literal

from dotenv import load_dotenv
load_dotenv(override=True)

# 로거 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

app = FastAPI(title="Agent_API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    )

@app.get("/health", tags=["Health"])
async def health_check():
    """Elastic Search Health Check"""
    logger.info("Health check requested")
    es = Elasticsearch("http://localhost:9200")
    try:
        if es.ping():
            logger.info("Elasticsearch is healthy")
            return JSONResponse(content={"status": "ok"})
        else:
            logger.error("Elasticsearch ping failed")
            return JSONResponse(status_code=500, content={"status": "error", "message": "Elasticsearch unreachable"})
    except Exception as e:
        logger.exception("Exception during Elasticsearch health check")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Router 등록
from routers.web import web_search
from routers.wiki import wiki_search, do_wiki_search
from routers.arxiv import arxiv_search
from routers.schedule import schedule
from routers.sql_agent import sql_agent
from routers.stream_agent import stream_agent
from routers.ppt_maker import pptx_maker
from routers.deep_agent import deep_agent
from routers.rag_agent import rag_agent

app.include_router(schedule)
app.include_router(rag_agent)
app.include_router(deep_agent)
app.include_router(web_search)
app.include_router(wiki_search)
app.include_router(arxiv_search)
app.include_router(sql_agent)
app.include_router(stream_agent)
app.include_router(pptx_maker)

# MCP 서버 생성
from fastapi_mcp import FastApiMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client

# 1. 외부 MCP 서버 설정 (JSON의 내용을 코드로 옮김)
# 1. Kiwi 항공권 검색 도구 추가 (SSE 방식)
@app.get("/search-flights", operation_id="search_flights")
async def search_flights(fly_from: str, fly_to: str, date_from: str):
    """
    항공권 정보를 검색하는 도구입니다.
    출발지(fly_from), 도착지(fly_to), 출발 날짜(date_from, DD/MM/YYYY 형식)가 필요합니다.
    예: '서울에서 도쿄 가는 비행기 찾아줘'라는 질문에 사용하세요.
    """
    # 외부 Kiwi MCP 서버 URL
    url = "https://mcp.kiwi.com" # 보통 /sse 경로를 사용합니다.
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 외부 MCP의 실제 도구 이름과 인자를 호출 (Kiwi MCP 명세를 따름)
            # 여기서는 예시로 'search_flights'라고 가정합니다. 
            # 실제 도구 이름은 Kiwi MCP 문서를 확인해야 합니다.
            result = await session.call_tool("search_flights", arguments={
                "fly_from": fly_from,
                "fly_to": fly_to,
                "date_from": date_from
            })
            
            if result.content:
                return result.content[0].text
            return "항공권 정보를 찾을 수 없습니다."


mcp = FastApiMCP(
    app,
    include_operations=[
        "wiki_search", 
        "web_search",
        "arxiv_search", 
        "search_flights"
        ],
    describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
    describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response
    )

# # FastAPI 앱에 MCP 서버 마운트
# mcp.mount_sse(app, mount_path="/mcp")

# class DynamicSearchRequest(BaseModel):
#     tool_name: Literal["wiki_search", "web_search", "arxiv_search", "search_flights"]
#     query: str


# @app.post("/dynamic_search", tags=["Tool"], operation_id="dynamic_search")
# def dynamic_search(req: DynamicSearchRequest):
#     logger.info(f"Dynamic search called: tool={req.tool_name}, query={req.query}")

#     if req.tool_name == "wiki_search":
#         return {"results": do_wiki_search(req.query)}

#     raise HTTPException(status_code=400, detail=f"Unsupported tool: {req.tool_name}")
    
# mcp = FastApiMCP(
#     app,
#     include_operations=["dynamic_search"],
#     describe_full_response_schema=True,
#     describe_all_responses=True,
# )

# mcp.mount_sse(app, mount_path="/mcp")

if __name__ == "__main__":
    
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)