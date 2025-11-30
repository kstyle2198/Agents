from fastapi import APIRouter, HTTPException

from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError

from functools import lru_cache
from langchain_ollama import OllamaEmbeddings


from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}")

class ElasticsearchIndexer:
    """
    Elasticsearch 연결, 인덱스 관리 및 데이터 색인 작업을 캡슐화하는 클래스입니다.
    """

    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "test_002"):
        """
        ElasticsearchIndexer 클래스를 초기화하고 Elasticsearch 연결 및 인덱스를 설정합니다.
        """
        self.INDEX_NAME = index_name
        self.es = Elasticsearch(es_url)  
        self.embedding_model = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest")
        self._ensure_index_exists()

    def get_elastic_index_list(self):
        response = self.es.cat.indices(index='*', format='json')
        return list(response)

    def search_documents(self, query_text: str = None, size: int = 5, min_score: float = 0.5):
        """
        Elasticsearch에서 텍스트 또는 임베딩을 기반으로 문서를 검색합니다.
        
        텍스트 검색 (query_text)과 벡터 검색 (query_embedding) 중 하나 또는 둘 다를 사용하여 검색할 수 있습니다.
        둘 다 제공되면 부스트 값이 적용된 Bool 쿼리 (RRF)를 구성합니다.
        
        Args:
            query_text (str, optional): 일반 텍스트 검색어. Defaults to None.
            query_embedding (list, optional): 벡터 검색을 위한 1024차원 임베딩 리스트. Defaults to None.
            size (int, optional): 반환할 최대 문서 수. Defaults to 10.
            min_score (float, optional): 최소 점수 임계값. Defaults to 0.5.
            
        Returns:
            list: 검색 결과 문서 (hit['_source'] + score) 리스트.
        """
        if not query_text:
            logger.warning("Either query_text or query_embedding must be provided.")
            return []

        search_body = {
            "size": size,
            "min_score": min_score,
            "query": {
                "bool": {
                    "should": [],
                    "minimum_should_match": 1 # 'should' 절 중 최소 하나는 일치해야 함
                }
            },
            # Elasticsearch 8.x 이상에서 kNN 검색을 위한 kNN 섹션 추가 (Elasticsearch 버전에 따라 달라질 수 있음)
            "knn": []
        }
        
        # 1. 일반 텍스트 검색 쿼리 (Query Text)
        if query_text:
            # "page_content" 필드에서 텍스트를 검색하는 match 쿼리 추가
            search_body["query"]["bool"]["should"].append({
                "match": {
                    "page_content": {
                        "query": query_text,
                        "boost": 1.0 # 텍스트 검색 부스트 값
                        }
                    }
                })
            logger.info(f"Text search enabled for: {query_text}")
        
            search_body["knn"].append({
                "field": "embeddings",
                "query_vector": self.embedding_model.embed_query(query_text),
                "k": size, # k: 이웃 수
                "num_candidates": max(size * 10, 50), # 검색할 후보 수 (성능/정확도 트레이드오프)
                "boost": 0.8 # 벡터 검색 부스트 값 (텍스트 검색보다 약간 낮게 설정)
                })
            logger.info("Vector search enabled.")
            
        try:
            # 3. 검색 실행
            res = self.es.search(
                index=self.INDEX_NAME, 
                body=search_body
            )
            
            # 4. 결과 파싱 및 반환
            hits = res['hits']['hits']
            
            # 결과에 점수 (Relevance Score)를 포함하여 반환합니다.
            documents = [{'_score': hit['_score'], **hit['_source']} for hit in hits]
            logger.info(f"Found {len(documents)} documents.")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

if __name__ == "__main__":
    

    pass



