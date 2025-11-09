### Logger 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

### LLM 설정
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv(override=True)
NO_THINK_MODEL = os.getenv("NO_THINK_MODEL")
reranking_model_path = os.getenv("RANK_MODEL_PATH")

llm = ChatGroq(model_name= NO_THINK_MODEL, temperature=0, max_retries=3, timeout=600)   


### 참고용 단어 변환 및 컬럼 설명 사전 로드
import pandas as pd
def get_ref_info(excel_path:str):
    """ Get reference info from excel file """
    # read excel
    df_convert = pd.read_excel(excel_path, sheet_name="convert")  # key word convert
    df_desc = pd.read_excel(excel_path, sheet_name="desc")   # column description
    # word change dict
    word_change = dict(zip(df_convert['before_word'], df_convert['after_word']))
    # desc dict
    desc_dict = dict(zip(df_desc['description'], df_desc['column_name']))
    return word_change, desc_dict

word_change, desc_dict = get_ref_info(excel_path="./data/schema_desc.xlsx")
word_change, desc_dict


#################################
### Refine Query ################
#################################
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, Dict, Optional, Union, TypedDict, Annotated

class GraphState(TypedDict):
    """State for query refinement"""
    question: str
    refined_query_first: str =""
    target_tables: List[str]
    sql_query: str
    query_result: str
    query_rows: list
    current_user: str
    attempts: int
    relevance: str
    sql_error: bool

def question_keyword_convert(question: str, word_change:dict) -> str:
    """ Convert keywords in the question based on the word_change dictionary """
    logger.debug(f"question_keyword_convert")
    for old_word, new_word in word_change.items():
        question = question.replace(old_word, new_word)
    logger.info(f"Keyword Converted question: {question}")
    return question

import re
def remove_think_tags(text):
    """
    <think> ~ </think> 태그와 그 내용을 제거합니다.
    """
    logger.debug(f"remove_think_tags")
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)

from langchain_core.messages import HumanMessage
def refine_first(state:GraphState):
    """Refine the question based on chat history"""
    logger.debug(f"refine_first")
    question = state['question']
    pre_refine_question = question_keyword_convert(question=question, word_change=word_change)

    try:
        prompt = f"""
        당신의 역할은 input 질문을 SQL 쿼리 변환에 적합하게 정제하는 것입니다.
        input 질문 내 영어 단어는 유지해주세요.
        아래 질문과 참고 정보를 활용하여 질문을 정제해주세요.
        질문의 대상이 되는 칼럼도 조회 항목에 포함해주세요.
        반드시 참고 정보에 포함된 내용에 기반하여 정제해주세요.
        정제된 질문만 출력해주세요.

        <question>
        {pre_refine_question}
        </question>

        <Reference Information>
        {str(desc_dict)}
        </Reference Information>

        refiend question: 
        """
        refined_question_content = llm.invoke([HumanMessage(content=prompt)]).content
        refined_question_content = remove_think_tags(refined_question_content)
        logger.info(f"refined_query_first: {refined_question_content}")
        return {"refined_query_first": refined_question_content}
    except Exception as e:
        logger.error(f"Error occurred during Refine - {e}")
        raise

def query_refiner(state):
    """Build the refinement graph"""
    refine_builder = StateGraph(state)
    refine_builder.add_node("refine_first", refine_first)

    refine_builder.add_edge(START, "refine_first")
    return refine_builder.compile()

sql_refine_graph = query_refiner(GraphState)

#################################
### SQL Search ################
#################################
## PostgreSQL 연결을 위한 DATABASE_URL
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost/mydb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

from sqlalchemy import text, inspect
def get_database_schema(target_tables: list[str]):
    """Retrieve the database schema for the specified tables."""
    logger.debug(f"get_database_schema")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost/mydb")
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        if table_name not in target_tables:
            continue
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name.upper()}: {col_type.upper()}\n"
        schema += "\n"
    logger.info("Retrieved database schema.")
    return schema

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
        )

def check_relevance(state: GraphState, config: RunnableConfig):
    logger.debug(f"check_relevance")
    target_tables = state["target_tables"]
    question = state["refined_query_first"]
    schema = get_database_schema(target_tables = target_tables)
    # print(f"Checking relevance of the question: {question}")
    system = """You are an assistant that determines whether a given question is related to the following database schema.

Schema:
{schema}

Respond with only "relevant" or "not_relevant".
""".format(schema=schema)
    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    state["relevance"] = relevance.relevance
    logger.info(f"Relevance determined: {state['relevance']}")
    return state

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the natural language question."
    )

def convert_nl_to_sql(state: GraphState, config: RunnableConfig):
    """ Convert natural language question to SQL query and update the state. """
    logger.debug(f"convert_nl_to_sql")  
    target_tables = state["target_tables"]
    question = state["refined_query_first"]
    schema = get_database_schema(target_tables = target_tables)
    # print(f"Converting question to SQL : {question}")
    system = f"""You are an assistant that converts natural language questions into SQL queries based on the following schema:

{schema}

Provide only the SQL query without any explanations. 
Alias columns appropriately to match the expected keys in the result.
Apply appropriate type castings when needed.

"""
    convert_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    structured_llm = llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    state["sql_query"] = result.sql_query
    logger.info(f"Generated SQL Query: {state['sql_query']}")
    return state

def format_query_result(data: list[dict]) -> str:
    """
    리스트[딕셔너리] 형태의 SQL Query 결과를 사람이 읽기 좋은 문자열로 변환.

    Args:
        data (list[dict]): Raw SQL Query 결과

    Returns:
        str: 포맷된 문자열
    """
    if not data:
        return "No data available."

    formatted_rows = []
    for row in data:
        # 각 key=value 형태로 변환
        formatted_row = ", ".join(f"{k}: {v}" for k, v in row.items())
        formatted_rows.append(formatted_row)

    return "\n".join(formatted_rows)

def execute_sql(state: GraphState):
    """ Execute the SQL query and update the state with results or errors. """
    logger.debug(f"execute_sql")
    sql_query = state["sql_query"].strip()
    session = SessionLocal()
    try:
        result = session.execute(text(sql_query))
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = result.keys()

            if rows:
                header = columns # ", ".join(columns)
                state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                # print(f"Raw SQL Query Result: {state['query_rows']}")

                # Format the result for readability
                formatted_result = format_query_result(state["query_rows"])
            else:
                state["query_rows"] = []
                formatted_result = "No results found."
            state["query_result"] = formatted_result
            state["sql_error"] = False
            logger.info(f"SQL SELECT query executed successfully. - {state["query_result"]}")
        else:
            session.commit()
            state["query_result"] = "The action has been successfully completed."
            state["sql_error"] = False
            logger.info(f"SQL command executed successfully. - {state["query_result"]}")
    except Exception as e:
        state["query_result"] = f"Error executing SQL query: {str(e)}"
        state["sql_error"] = True
        logger.error(f"Error executing SQL query: {str(e)}")
    finally:
        session.close()
    return state

def generate_human_readable_answer(state: GraphState):
    """ Generate a human-readable answer based on the SQL query and its result. """
    logger.debug(f"generate_human_readable_answer")
    question = state["refined_query_first"]
    sql = state["sql_query"]
    result = state["query_result"]
    query_rows = state.get("query_rows", [])
    sql_error = state.get("sql_error", False)

    system = f"""
    당신은 경영 전략을 수립하는 기획 전문가 입니다.
    아래 질문과 참고 정보를 활용하여 최고경영 임원에게 제공할 경영분석 보고서를 작성해주세요.
    반드시 참고 정보에 근거하여 답변을 생성해주세요.
    """


    if sql_error:
        # Directly relay the error message
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"""Question:
{question}

Result:
{str(result)}

Formulate a clear and understandable error message in a single sentence, informing them about the issue.

"""
                ),
            ]
        )
    elif sql.lower().startswith("select"):
        if not query_rows:
            # Handle cases with no orders
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        f"""Question:
{question}

Result:
{str(result)}

검색 데이터는 누락없이 표 형식으로 정리해주세요.
반드시 참고 정보에 포함된 내용에 기반하여 작성해주세요.
한글로 작성하되, 원문에 영어 단어로 표기된 부분은 그대로 유지해주세요."""
                    ),
                ]
            )
        else:
            # Handle displaying orders
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        f"""question:
{question}

Result:
{str(result)}

검색 데이터는 누락없이 표 형식으로 정리해주세요.
반드시 참고 정보에 포함된 내용에 기반하여 작성해주세요.
한글로 작성하되, 원문에 영어 단어로 표기된 부분은 그대로 유지해주세요."""
                    ),
                ]
            )
    else:
        # Handle non-select queries
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"""question:
{question}

Result:
{str(result)}

검색 데이터는 누락없이 표 형식으로 정리해주세요.
반드시 참고 정보에 포함된 내용에 기반하여 작성해주세요.
한글로 작성하되, 원문에 영어 단어로 표기된 부분은 그대로 유지해주세요."""
                ),
            ]
        )
    
    human_response = generate_prompt | llm | StrOutputParser()
    answer = human_response.invoke({})
    state["query_result"] = answer
    logger.info(f"Generated human-readable answer. - {state["query_result"]}")
    return state

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

def regenerate_query(state: GraphState):
    """ Regenerate the SQL query by rewriting the question to ensure all necessary details are included. """
    logger.debug(f"regenerate_query")
    question = state["refined_query_first"]
    system = """You are an assistant that reformulates an original question to enable more precise SQL queries. 
    Ensure that all necessary details, such as table joins, are preserved to retrieve complete and accurate data.
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                f"Original Question: {question}\nReformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved.",
            ),
        ]
    )
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({})
    state["refined_query_first"] = rewritten.question
    state["attempts"] += 1
    logger.info(f"Rewritten question: {state['refined_query_first']}")
    return state

def relevance_router(state: GraphState):
    """ Route based on the relevance of the question to the database schema. """
    if state["relevance"].lower() == "relevant":
        return "convert_to_sql"
    else:
        return "no_relevance"

def execute_sql_router(state: GraphState):
    """ Route based on whether the SQL execution resulted in an error. """
    if not state.get("sql_error", False):
        return "generate_human_readable_answer"
    else:
        return "regenerate_query"

def check_attempts_router(state: GraphState):
    """ Route based on the number of attempts made to refine the query. """
    if state["attempts"] < 3:
        return "convert_to_sql"
    else:
        return "end_max_iterations"

def end_max_iterations(state: GraphState):
    """ Handle the scenario when maximum attempts are reached. """
    state["query_result"] = "Please try again."
    logger.debug("Maximum attempts reached. Ending the workflow.")
    return state

import json
from langgraph.checkpoint.memory import MemorySaver
def sql_builder(state):
    """ Build the SQL processing graph. """
    sql_builder = StateGraph(state)
    sql_builder.add_node("sql_refine", sql_refine_graph)
    sql_builder.add_node("check_relevance", check_relevance)
    sql_builder.add_node("convert_to_sql", convert_nl_to_sql)
    sql_builder.add_node("execute_sql", execute_sql)
    sql_builder.add_node("generate_human_readable_answer", generate_human_readable_answer)
    sql_builder.add_node("regenerate_query", regenerate_query)
    sql_builder.add_node("end_max_iterations", end_max_iterations)

    sql_builder.add_edge(START, "sql_refine")
    sql_builder.add_edge("sql_refine", "check_relevance")
    sql_builder.add_conditional_edges(
        "check_relevance",
        relevance_router,
        {
            "convert_to_sql": "convert_to_sql",
            "no_relevance": END,
        },
    )
    sql_builder.add_edge("convert_to_sql", "execute_sql")
    sql_builder.add_conditional_edges(
        "execute_sql",
        execute_sql_router,
        {
            "generate_human_readable_answer": "generate_human_readable_answer",
            "regenerate_query": "regenerate_query",
        },
    )

    sql_builder.add_conditional_edges(
        "regenerate_query",
        check_attempts_router,
        {
            "convert_to_sql": "convert_to_sql",
            "max_iterations": "end_max_iterations",
        },
    )

    sql_builder.add_edge("generate_human_readable_answer", END)
    sql_builder.add_edge("end_max_iterations", END)
    memory_saver = MemorySaver()
    return sql_builder.compile(checkpointer=memory_saver)

sql_agent_graph = sql_builder(GraphState)

from uuid import uuid4
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from langgraph.errors import GraphRecursionError
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessageChunk

sql_agent = APIRouter()

class SqlAgentRequest(BaseModel):
    """Request model for query refinement"""
    question: str
    refined_query_first: str  = ""
    target_tables: List[str]
    attempts: int = 0
    thread_id: str

class SqlAgentResponse(BaseModel):
    """Response model for query refinement"""
    sql_query: str
    query_rows: list
    current_user: str
    attempts: int = 0
    relevance: str
    sql_error: bool
    session_id: str
    query_result: str
    history: list


@sql_agent.post("/astream", tags=["sql_agent"])
async def chat_stream(input: SqlAgentRequest):
    return StreamingResponse(
        generate_chat_responses(
            question=input.question,
            target_tables=input.target_tables,
            attempts=input.attempts,
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
    
async def generate_chat_responses(question: str, target_tables:list, attempts:int, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        inputs = {
            "question": question,
            "target_tables": target_tables,
            "attempts": attempts,
            }

        config = {
            "recursion_limit": 20,
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }

        # Initialize with first message
        events = sql_agent_graph.astream_events(inputs, version="v2", config=config)

        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        inputs = {
            "question": question,
            "target_tables": target_tables,
            "attempts": attempts,
            }
        
        config = {
            "recursion_limit": 20,
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = sql_agent_graph.astream_events(inputs, version="v2", config=config)

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

@sql_agent.get("/sql_threads", tags=["sql_agent"])
def list_threads(thread_id: str):
    try:
        # MemorySaver에 저장된 모든 스레드 목록 조회
        threads = list(sql_agent_graph.get_state_history(config={"configurable": {"thread_id": thread_id}}))
        logger.info(f"Retrieved history for thread_id {thread_id}: {threads}")
        # 임시 응답 (실제 구현에 맞게 수정 필요)
        return {"threads": threads}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @sql_agent.post("/astream", tags=["SQL_Agents"], operation_id="sql_agent_astream")
# async def app_astream_endpoint(request: SqlAgentRequest):
#     """
#     FastAPI endpoint wrapping the async graph.astream() function.

#     Args:
#         request: StreamRequest object containing target_tables, question, etc.

#     Returns:
#         JSON containing the final output value or error message.
#     """
#     thread_id = f"thread-{request.session_id}"
#     inputs = {
#         "question": request.question,
#         "target_tables": request.target_tables,
#         "attempts": request.attempts,
#         }
#     config = {
#         "recursion_limit": 20,
#         "configurable": {"thread_id": thread_id},
#         }

#     try:
#         value = None
#         async for output in sql_agent_graph.astream(inputs, config):
#             for key, val in output.items():
#                 # print(f">>> Node : {key}")
#                 value = val  # 마지막 값만 저장
#             # print("=" * 70)

#         if value is None:
#             raise HTTPException(status_code=500, detail="No output generated")

#         logger.info(f"final output: \n{value}")

#         history = list(sql_agent_graph.get_state_history(config={"configurable": {"thread_id": thread_id}}))
#         logger.info(f"history: \n{history}")
#         return {"final_output": value, "history": history}

#     except GraphRecursionError:
#         error_msg = f"=== Recursion Error - {request.recursion_limit} ==="
#         logger.error(error_msg)
#         raise HTTPException(status_code=500, detail=error_msg)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# class SQLRefineRequest(BaseModel):
#     """Request model for query refinement"""
#     question: str

# class SQLRefineResponse(BaseModel):
#     """Response model for query refinement"""
#     refined_query_first: str

# @sql_agent.post("/sql_refine", response_model=SQLRefineResponse, tags=["SQL_Agents"], operation_id="sql_refine_question")
# async def refine_question(req: SQLRefineRequest):
#     logger.info(f"Refine the query: {req.question}")
#     try:
#         # 초기 상태 구성
#         state = {
#             "question": req.question,
#             }

#         # 그래프 실행
#         result = sql_refine_graph.invoke(state)
#         logger.info(f"Refining Query completed. - {result["refined_query_first"]}")
#         return SQLRefineResponse(refined_query_first=result["refined_query_first"])

#     except Exception as e:
#         logger.error("Query Refinery failed", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))