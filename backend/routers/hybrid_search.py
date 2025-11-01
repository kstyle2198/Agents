import uuid
import heapq
import operator
from FlagEmbedding import FlagReranker
from typing import List, Dict, Optional, Union, TypedDict, Annotated
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from langchain_ollama import OllamaEmbeddings
# from langchain_elasticsearch import ElasticsearchStore
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from transformers import AutoModel
from elasticsearch import Elasticsearch
from functools import lru_cache

# 로거 설정
import logging
from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}", level=logging.DEBUG)

import os
from dotenv import load_dotenv
load_dotenv(override=True)
think_model = os.getenv("THINK_MODEL")
no_think_model = os.getenv("NO_THINK_MODEL")

reranker = FlagReranker(model_name_or_path="../llms/bge-reranker-v2-m3" , 
                        use_fp16=True,
                        batch_size=512,
                        max_length=2048,
                        normalize=True)

# ✅ 1. 임베딩 모델 캐싱
@lru_cache(maxsize=1)
def get_ollama_embedding():
    logger.info("Initializing Ollama Embeddings (cached)")
    return OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="bge-m3:latest"
    )

# ✅ 2. 엘라스틱 클라이언트 캐싱
@lru_cache(maxsize=1)
def get_elastic_client():
    logger.info("Connecting to Elasticsearch (cached)")
    return Elasticsearch(
        "http://localhost:9200",
        basic_auth=("Kstyle", "12345"),
        verify_certs=False,
        request_timeout=5,
        max_retries=1,
    )

class ElasticsearchVectorStore:
    """Elasticsearch 벡터 저장소 직접 구현"""
    
    def __init__(self, index_names: Union[str, List[str]], embedding_model):
        self.es_client = get_elastic_client()
        self.embedding_model = embedding_model
        
        if isinstance(index_names, str):
            index_names = [index_names]
        self.index_names = index_names
        
        # 인덱스 존재 여부 확인
        existing_indices = self.es_client.indices.get_alias(index="*").keys()
        self.valid_indices = [i for i in index_names if i in existing_indices]
        
        if not self.valid_indices:
            raise ValueError(f"No valid indices found in Elasticsearch: {index_names}")
        
        logger.info(f"Initialized ElasticsearchVectorStore with indices: {self.valid_indices}")
    
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Dict]:
        """쿼리와 유사한 문서 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.embed_query(query)
            
            # Elasticsearch에서 벡터 유사도 검색
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": k,
                "_source": ["content", "metadata"]
            }
            
            results = []
            for index_name in self.valid_indices:
                try:
                    response = self.es_client.search(
                        index=index_name,
                        body=search_body
                    )
                    
                    for hit in response["hits"]["hits"]:
                        result = {
                            "content": hit["_source"].get("content", ""),
                            "metadata": hit["_source"].get("metadata", {}),
                            "score": hit["_score"]
                        }
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error searching index {index_name}: {e}")
                    continue
            
            # 점수 기준으로 정렬 및 상위 k개 결과 반환
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in similarity_search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple]:
        """점수와 함께 유사도 검색 수행"""
        results = self.similarity_search(query, k, **kwargs)
        return [(result["content"], result["score"]) for result in results]
    
    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20, **kwargs) -> List[str]:
        """MMR(Maximal Marginal Relevance) 검색"""
        try:
            # 더 많은 결과를 먼저 가져옴
            initial_results = self.similarity_search(query, fetch_k, **kwargs)
            
            if not initial_results:
                return []
            
            # 쿼리 임베딩
            query_embedding = self.embedding_model.embed_query(query)
            
            # 결과 임베딩 계산 (간단한 구현)
            results_with_embeddings = []
            for result in initial_results:
                # 실제 구현에서는 문서의 임베딩을 저장해두거나 다시 계산
                doc_embedding = self.embedding_model.embed_query(result["content"])
                results_with_embeddings.append({
                    "content": result["content"],
                    "embedding": doc_embedding,
                    "score": result["score"]
                })
            
            # MMR 선택
            selected = []
            remaining = results_with_embeddings.copy()
            
            # 첫 번째 문서는 가장 유사한 것으로 선택
            if remaining:
                selected.append(remaining.pop(0))
            
            # 나머지 문서 선택
            while len(selected) < k and remaining:
                best_score = -1
                best_idx = -1
                
                for i, doc in enumerate(remaining):
                    # 다양성 점수 계산 (간단한 구현)
                    similarity_to_query = self._cosine_similarity(query_embedding, doc["embedding"])
                    
                    max_similarity_to_selected = 0
                    for sel in selected:
                        sim = self._cosine_similarity(sel["embedding"], doc["embedding"])
                        if sim > max_similarity_to_selected:
                            max_similarity_to_selected = sim
                    
                    # MMR 점수
                    mmr_score = similarity_to_query - max_similarity_to_selected
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))
            
            return [doc["content"] for doc in selected]
            
        except Exception as e:
            logger.error(f"Error in MMR search: {e}")
            # 실패시 일반 유사도 검색으로 폴백
            results = self.similarity_search(query, k, **kwargs)
            return [result["content"] for result in results]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_elastic_vectorstore(index_names: Union[str, List[str]]):
    """Elasticsearch 벡터 저장소 로드"""
    logger.info("Load Elastic VectorStore")

    try:
        embedding = get_ollama_embedding()
        es_client = get_elastic_client()

        # ✅ 인덱스 존재 여부 확인
        existing_indices = es_client.indices.get_alias(index="*").keys()
        valid_indices = [i for i in ([index_names] if isinstance(index_names, str) else index_names) 
                        if i in existing_indices]

        if not valid_indices:
            raise ValueError(f"No valid indices found in Elasticsearch: {index_names}")

        # ✅ 직접 구현한 ElasticsearchVectorStore 사용
        vector_store = ElasticsearchVectorStore(
            index_names=valid_indices,
            embedding_model=embedding
        )

        logger.info(f"Loaded {len(valid_indices)} existing indices successfully.")
        return vector_store

    except Exception:
        logger.exception("Error occurred during Elastic VectorStore Loading")
        raise


# def load_elastic_vectorstore(index_names: Union[str, List[str]]):
#     logger.info("Load Elastic VectorStore")

#     try:
#         if isinstance(index_names, str):
#             index_names = [index_names]

#         embedding = get_ollama_embedding()
#         es_client = get_elastic_client()

#         # ✅ 인덱스 존재 여부를 사전에 확인 (불필요한 생성 방지)
#         existing_indices = es_client.indices.get_alias(index="*").keys()
#         valid_indices = [i for i in index_names if i in existing_indices]

#         if not valid_indices:
#             raise ValueError(f"No valid indices found in Elasticsearch: {index_names}")

#         # ✅ ElasticsearchStore 초기화 (불필요한 인덱스 생성 방지)
#         vector_store = ElasticsearchStore(
#             index_name=valid_indices,
#             embedding=embedding,
#             es_connection=es_client
#         )

#         logger.info(f"Loaded {len(valid_indices)} existing indices successfully.")
#         return vector_store

#     except Exception:
#         logger.exception("Error occurred during Elastic VectorStore Loading")
#         raise

# def load_elastic_vectorstore(index_names: Union[str, List[str]]):
#     logger.info(f"Load Elastic VectorStore")
#     try:
#         # 단일 문자열인 경우 리스트로 변환
#         if isinstance(index_names, str):
#             index_names = [index_names]
        
#         vector_store = ElasticsearchStore(
#             index_name=index_names, 
#             embedding=OllamaEmbeddings(
#                 base_url="http://localhost:11434", 
#                 model="bge-m3:latest"
#             ), 
#             es_url="http://localhost:9200",
#             es_user="Kstyle",
#             es_password="12345",
#         )
#         return vector_store
#     except Exception as e:
#         logger.exception("Error occurred during Elastic VectorStore Loading")
#         raise

index_names = ["rule"]
vector_store = load_elastic_vectorstore(index_names=index_names)

def reranking(query: str, docs: list, min_score: float = 0.5, top_k: int = 3):
    """
    Reranks documents based on a query.
    """
    logger.info(f"Start ReRanking")
    try:
        inputs = [[query.lower(), doc["page_content"].lower()] for doc in docs]
        scores = reranker.compute_score(inputs)
        if not isinstance(scores, list):
            scores = [scores]
        logger.info(f"---- original scores: {scores}")
        # Filter scores by threshold and keep index
        filtered_scores = [(score, idx) for idx, score in enumerate(scores) if score >= min_score]
        # Get top_k using heapq (more efficient than sorting full list)
        top_scores = heapq.nlargest(top_k, filtered_scores, key=lambda x: x[0])
        # Get document objects from top indices
        reranked_docs = [docs[idx] for _, idx in top_scores]
        logger.info(f"ReRanking completed. Found {len(reranked_docs)} results.")
        return top_scores, reranked_docs
    except Exception as e:
        logger.error("Error occurred during ReRanking")
        raise

class OverAllState(TypedDict):
    """
    Represents the overall state of our graph.
    """
    question: str
    questions: Annotated[list, operator.add] # 질문 누적
    context: Annotated[list, operator.add] # 검색 결과 누적
    rerank_context: Annotated[list, operator.add] # 리랭크된 검색 결과 누적
    top_scores: Annotated[list, operator.add] # 리랭크 점수 누적
    top_k: Dict = {'doc':4, "web": 2}
    rerank_k: int = 3
    rerank_threshold: float = 0.01
    generations: Annotated[list, operator.add] # 응답 결과 누적

def retrieve_agent(state: OverAllState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, 
        that contains retrieved documents
    """
    logger.info(f"Start Retrieve Agent")
    try:
        question = state['question']
        top_k_doc = state['top_k']['doc']
        
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"fetch_k": 10, "k":top_k_doc},
            )
        documents = retriever.invoke(question)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]

        logger.info(f"Retrieving Docs completed. Found {len(documents)} results.")
        # 검색 결과를 context에 추가
        return {"context": [documents]}
    except Exception as e:
        logger.exception("Error occurred during retrieve agent")
        raise

def reranking_agent(state:OverAllState):
    """Rerank retrieved documents"""
    logger.info(f"Start Reranking Agent")
    try:
        question = state['question']
        context = state['context'][0]
        rerank_k = state['rerank_k']
        rerank_threshold = state["rerank_threshold"]
        
        top_scores, documents = reranking(query=question, docs=context, min_score = rerank_threshold, top_k= rerank_k)
        logger.info(f"---- Reranking 문서개수: {len(documents)} / top_scores: {top_scores}")
        
        # 리랭크된 문서와 스코어를 각각의 리스트에 추가
        logger.info(f"Reranking Agent completed. Rerank {len(documents)} results.")
        return {"rerank_context": [documents], "top_scores": [top_scores]}
    except Exception as e:
        logger.exception("Error occurred during Reranking Agent")
        raise

def search_builder(state):
    rag_builder = StateGraph(state)
    rag_builder.add_node("retrieve_agent", retrieve_agent)
    rag_builder.add_node("reranking_agent", reranking_agent)

    rag_builder.add_edge(START, "retrieve_agent") 
    rag_builder.add_edge("retrieve_agent", "reranking_agent")
    rag_builder.add_edge("reranking_agent", END) 

    memory = MemorySaver()
    return rag_builder.compile(checkpointer=memory)

search_graph = search_builder(OverAllState)

# Define request models
class QuestionRequest(BaseModel):
    question: str
    top_k: Dict = {'doc':3, 'web':2}
    rerank_k: Optional[int] = 2
    rerank_threshold: Optional[float] = 0.0
    session_id: str = None  # 클라이언트가 줄 수도 있고 안 줄 수도 있음

class SearchResponse(BaseModel):
    refined_question: str
    documents: List[Dict]
    scores: List[float]
    questions_history: List[str] # 누적된 질문 히스토리 추가
    search_results_history: List[List[Dict]] # 누적된 검색 결과 히스토리 추가 (각 검색 단계별 결과 리스트)
    reranked_results_history: List[List[Dict]] # 누적된 리랭크 결과 히스토리 추가 (각 리랭크 단계별 결과 리스트)
    rerank_scores_history: List[List[float]] # 누적된 리랭크 스코어 히스토리 추가 (각 리랭크 단계별 스코어 리스트)

hybrid_search = APIRouter()

@hybrid_search.post("/hybrid_search", response_model=SearchResponse, tags=["Tool"], operation_id="safety_search")
async def search_documents(request: QuestionRequest):
    """
    Endpoint to search documents based on a question.
    Performs retrieval from vector store and web search, then reranks results.
    """
    logger.info(f"Hybrid search API called")
    try:
        # ✅ session_id 자동 생성 또는 재사용
        session_id = request.session_id or str(uuid.uuid4())
        thread_id = f"thread-{session_id}"
        
        # Initialize state with default values and empty lists for accumulation
        state = {
            "question": request.question,
            "top_k": request.top_k,
            "rerank_k": request.rerank_k,
            "rerank_threshold": request.rerank_threshold,
            "questions": [], # 여기에 첫 질문이 refined_question에서 추가됨
            "context": [],
            "rerank_context": [],
            "top_scores": [],
            "generations": [], # /search에서는 사용하지 않지만, state 정의에 포함
        }
        
        # Execute the search graph
        # ainvoke는 최종 상태를 반환합니다.
        search_result = await search_graph.ainvoke(state, config={"thread_id": thread_id})

        print("-----search result -----")        
        # Prepare response
        documents = []
        # rerank_context는 리스트의 리스트이므로, 가장 마지막 리랭크 결과를 가져옵니다.
        if search_result["rerank_context"]:
            documents = search_result["rerank_context"][-1] 

        scores = []
        # top_scores도 리스트의 리스트이므로, 가장 마지막 리랭크 스코어를 가져옵니다.
        if search_result["top_scores"]:
            scores = [score[0] for score in search_result["top_scores"][-1]]
        
        return SearchResponse(
            refined_question = search_result["question"],
            documents=documents,
            scores=scores,
            questions_history=search_result["questions"], # 누적된 질문 히스토리
            search_results_history=search_result["context"], # 누적된 검색 결과 히스토리
            reranked_results_history=search_result["rerank_context"], # 누적된 리랭크 결과 히스토리
            rerank_scores_history=[[s[0] for s in score_list] for score_list in search_result["top_scores"]] # 누적된 리랭크 스코어 히스토리
        )
    
    except Exception as e:
        logger.error("Hybrid search failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")