from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deep_research.agent import agent


from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}")

from dotenv import load_dotenv
load_dotenv()


# Router 구성
deep_agent = APIRouter()

class QueryRequest(BaseModel):
    query: str

@deep_agent.post("/deep-agent", tags=["Deep_Agent"])
async def run_deep_agent(request: QueryRequest):
    """
    Deep Agent에게 query를 보내고 결과를 반환하는 엔드포인트
    """
    try:
        logger.info(request.query)
        input_message = {"messages": [{"role": "user","content": request.query,}],}
        result = await agent.ainvoke(input_message)
        logger.info(result)

        # 여기서 문자열 결과를 dict로 강제 변환
        if isinstance(result, str):
            result = {"output": result}

        # LangGraph의 최종 상태도 dict 내에서 문자열이 있을 수 있음
        if "output" in result and isinstance(result["output"], str):
            result = {"output": result["output"]}

        return {"status": "success", "response": result}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))