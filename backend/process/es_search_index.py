from elasticsearch import Elasticsearch
from langchain_ollama import OllamaEmbeddings

from utils.setlogger import setup_logger
logger = setup_logger(f"{__name__}")



class ElasticsearchIndexer:
    """
    Elasticsearch 연결, 인덱스 관리 및 데이터 색인 작업을 캡슐화하는 클래스입니다.
    """

    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "open_paper"):
        """
        ElasticsearchIndexer 클래스를 초기화하고 Elasticsearch 연결 및 인덱스를 설정합니다.
        """
        self.INDEX_NAME = index_name
        self.es = Elasticsearch(es_url)
        self.embedding_model = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest")


# 👇️ 요청하신 검색 메서드 추가
    def search_documents(self, query_text: str = None, size: int = 10, min_score: float = 0.5):
        """
        Elasticsearch에서 텍스트 또는 임베딩을 기반으로 문서를 검색합니다.
        
        텍스트 검색 (query_text)과 벡터 검색 (query_embedding) 중 하나 또는 둘 다를 사용하여 검색할 수 있습니다.
        둘 다 제공되면 부스트 값이 적용된 Bool 쿼리 (RRF)를 구성합니다.
        
        Args:
            query_text (str, optional): 일반 텍스트 검색어. Defaults to None.
            size (int, optional): 반환할 최대 문서 수. Defaults to 10.
            min_score (float, optional): 최소 점수 임계값. Defaults to 0.5.
            
        Returns:
            list: 검색 결과 문서 (hit['_source'] + score) 리스트.
        """
        if not query_text:
            logger.warning("Either query_text  must be provided.")
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
        
        # 2. 벡터 검색 쿼리 (Query Embedding)            
        # kNN 섹션에 dense_vector 검색 추가
        # 참고: Elasticsearch 8.x 버전에서는 search API의 'knn' 파라미터를 사용하거나
        # 7.x 버전에서는 'script_score' 쿼리를 사용할 수 있습니다.
        # 여기서는 8.x의 'knn' 파라미터를 사용하는 표준 방식을 따릅니다.
        search_body["knn"].append({
            "field": "embeddings",
            "query_vector": self.embedding_model.embed_query(query_text),
            "k": size, # k: 이웃 수
            "num_candidates": max(size * 10, 50), # 검색할 후보 수 (성능/정확도 트레이드오프)
            "boost": 0.8 # 벡터 검색 부스트 값 (텍스트 검색보다 약간 낮게 설정)
        })
        
        # kNN을 사용할 경우, 최소 점수 대신 필터링을 사용하여 관련 없는 문서를 제거할 수 있습니다.
        # 이 예시에서는 min_score를 유지합니다.
        
        logger.info("Vector search enabled.")
            
        try:
            # 3. 검색 실행
            # Elasticsearch 8.x에서는 kNN과 쿼리를 조합할 수 있습니다.
            # 'knn' 파라미터가 비어 있지 않으면 'search_body'에서 'knn'을 제거하고 별도의 'knn' 인수로 전달해야 합니다.
            # 하지만 8.x 클라이언트의 search 메서드가 body에 knn을 허용하는 경우가 많으므로 body에 포함합니다.
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
        
    def get_all_index_names(self):
        """
        Elasticsearch 클러스터에 존재하는 모든 인덱스의 이름을 리스트로 반환합니다.
        """
        try:
            # indices.get_alias("*")는 모든 인덱스의 별칭 정보를 가져오며, 
            # 딕셔너리의 키(key)가 인덱스 이름입니다.
            indices_dict = self.es.indices.get_alias(index="*")
            index_names = list(indices_dict.keys())
            
            logger.info(f"Retrieved {len(index_names)} indices from Elasticsearch.")
            return indices_dict
            
        except Exception as e:
            logger.error(f"Error fetching all index names: {e}")
            return []

es = ElasticsearchIndexer(index_name="open_paper")

if __name__ == "__main__":

    # index_list = es.get_all_index_names()
    # print(index_list)

    # res = es.search_documents(query_text="how to make text embedding")
    # print(res)
    pass