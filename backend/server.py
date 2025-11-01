import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from langchain_ollama import OllamaEmbeddings

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
from routers.wiki import wiki_search
from routers.arxiv import arxiv_search
# from routers.hybrid_search import hybrid_search
from routers.refine import refine
from routers.generate import generate
from routers.schedule import schedule
from routers.sql_agent import sql_agent
from routers.stream_agent import stream_agent

app.include_router(schedule)
app.include_router(refine)
# app.include_router(hybrid_search)
app.include_router(web_search)
app.include_router(wiki_search)
app.include_router(arxiv_search)
app.include_router(generate)
app.include_router(sql_agent)
app.include_router(stream_agent)


def load_elastic_vectorstore(index_names):
    logger.info("Initializing Elasticsearch vector store...")
    if isinstance(index_names, str):
        index_names = [index_names]

    return ElasticsearchStore(
        index_name=index_names,
        embedding=OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="bge-m3:latest"
        ),
        es_url="http://localhost:9200",
        es_user=os.getenv("ES_USER", "Kstyle"),
        es_password=os.getenv("ES_PASSWORD", "12345"),
    )

# MCP 서버 생성
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(
    app,
    include_operations=["wiki_search", "web_search", "arxiv_search"],
    describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
    describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response
    )

# FastAPI 앱에 MCP 서버 마운트
mcp.mount_sse(app, mount_path="/mcp")

if __name__ == "__main__":
    

    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)