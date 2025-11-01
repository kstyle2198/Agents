from langchain_tavily import TavilySearch
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(override=True)

# 로거 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

def do_web_search(question: str):
    logger.info(f"Performing web search for question: {question}")
    try:
        web_search_tool1 = TavilySearch(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        web_results = web_search_tool1.invoke({'query': question})
        logger.info(f"Web search completed. Found {len(web_results.get('results', []))} results.")
        return web_results
    except Exception as e:
        logger.exception("Error occurred during web search")
        raise


from pydantic import BaseModel
class SearchRequest(BaseModel):
    query: str


from fastapi import APIRouter, HTTPException
web_search = APIRouter()

@web_search.post("/web_search", tags=["Tool"], operation_id="web_search")
def do_search(request: SearchRequest):
    logger.info(f"Web search API called with query: {request.query}")
    try:
        results = do_web_search(request.query)
        return {"results": results}
    except Exception as e:
        logger.error("Web search failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

