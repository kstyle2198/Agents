import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}")

load_dotenv()

def web_search_tool(query: str):
    """Search on web for given query using tavily search tool"""
    logger.info(f"Performing web search for question: {query}")
    try:
        web_search_tool = TavilySearch(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        web_results = web_search_tool.invoke(query)
        logger.info(f"Web search completed. Found {len(web_results.get('results', []))} results.")
        return web_results
    except Exception as e:
        logger.exception("Error occurred during web search")
        raise


tools = [web_search_tool]

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
think_model = os.getenv("THINK_MODEL")
llm = ChatGroq(model_name=think_model, temperature=0,max_tokens=3000,) 
memory = MemorySaver()
llm_with_tools = llm.bind_tools(tools=tools)


# 프롬프트 정의
SYSTEM_PROMPT = """
You are the Smart AI Assistant in a company.
Based on the result of tool calling, Generate a consice and logical answer.
and if there is no relevant infomation in the tool calling result, Just say 'I don't know'.
Answer in Korean.
"""

async def agent_node(state: GraphState):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["messages"]
    ]
    result = await llm_with_tools.ainvoke(messages )
    return {"messages": [result],}

tool_node = ToolNode(tools=tools)

def get_graph():
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile(checkpointer=memory)

# Create Graph Object
graph = get_graph()

from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessageChunk
from pydantic import BaseModel, Field

# Create FastAPI Object
stream_agent = APIRouter()

class InputData(BaseModel):
    query: str
    thread_id: str = Field(default="001")

@stream_agent.post("/chat", tags=["Stream_Agent"])
async def chat_stream(input: InputData):
    return StreamingResponse(
        generate_chat_responses(
            message=input.query,
            checkpoint_id=input.thread_id
        ),
        media_type="text/event-stream"
        )

def serialise_ai_message_chunk(chunk):
    if (isinstance(chunk, AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )
    
async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }

        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # print("messages in else: ", message)
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")

            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for search
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search"]

            if search_calls:
                # Signal that a search is starting
                search_query = search_calls[0]["args"].get("query", "")
                # Escape quotes and special characters
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "tavily_search":
            # Search completed - send results or error
            output = event["data"]["output"]
            results = output["results"]
            logger.info("results_list: ", results)

            # Check if output is a list
            if isinstance(results, list):
                # Extract URLs from list of search results
                urls = []
                for item in results:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])

                # Convert URLs to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"

    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"

@stream_agent.get("/threads", tags=["Stream_Agent"])
def list_threads(thread_id: str):
    try:
        # MemorySaver에 저장된 모든 스레드 목록 조회
        threads = list(graph.get_state_history(config={"configurable": {"thread_id": thread_id}}))
        logger.info(f"Retrieved history for thread_id {thread_id}: {threads}")
        # 임시 응답 (실제 구현에 맞게 수정 필요)
        return {"threads": threads}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))