import json
import pickle
import psycopg2
from psycopg2.extras import Json
from psycopg2.extras import execute_batch
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from uuid import uuid4
import sys
from pathlib import Path
utils_path = Path(__file__).parent.parent
sys.path.append(str(utils_path))

from utils.config import get_config
from utils.setlogger import setup_logger
config = get_config()
logger = setup_logger(f"{__name__}", level=config.LOG_LEVEL)

class PostgresPipeline:
    def __init__(self, host="localhost", database="mydb", user=config.POSTGRES_USER, password=config.POSTGRES_PW):
        """데이터베이스 연결 정보를 초기화합니다."""
        self.db_config = {
            "host": host,
            "database": database,
            "user": user,
            "password": password
        }

    def _get_db_connection(self):
        """데이터베이스 연결을 반환합니다."""
        try:
            logger.info("START - POSTGRESS DB CONNECTION")
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"CONNECTION ERROR: {e}")
            raise

    def get_all_tables(self):
        """데이터베이스의 모든 테이블 이름을 조회합니다."""
        conn = None
        try:
            logger.info("START - GET TABLE NAMES")
            # DB 연결
            conn = self._get_db_connection()
            cur = conn.cursor()

            # 모든 테이블 이름 조회 (시스템 테이블 제외)
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
            """
            
            cur.execute(query)
            tables = cur.fetchall()

            return [table[0] for table in tables]

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error("오류 발생:", error)
            return []
        finally:
            if conn is not None:
                cur.close()
                conn.close()

    def drop_table(self, table_name: str):
        """지정된 테이블을 삭제합니다."""
        conn = None
        try:
            logger.info("DROP TABLE")
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # 테이블 삭제 SQL 실행
            drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
            cursor.execute(drop_query)
            conn.commit()
            
            logger.info(f"테이블 '{table_name}'이(가) 성공적으로 삭제되었습니다.")
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"테이블 삭제 오류: {e}")
            raise
        finally:
            if conn:
                cursor.close()
                conn.close()

    def create_table(self, table_name: str, columns_config: List[Dict[str, str]]):
        """
        PostgreSQL 데이터베이스에 새로운 테이블을 생성합니다.
        
        Args:
            table_name (str): 생성할 테이블 이름
            columns_config (List[Dict]): 컬럼 구성 정보
                예: [
                    {"name": "id", "type": "SERIAL PRIMARY KEY"},
                    {"name": "name", "type": "VARCHAR(100) NOT NULL"},
                    {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"}
                ]
        """
        conn = None
        try:
            logger.info("CREATE TABLE")
            # 데이터베이스 연결
            conn = self._get_db_connection()
            cur = conn.cursor()

            # 컬럼 정의 생성
            column_definitions = []
            for column in columns_config:
                column_definitions.append(f"{column['name']} {column['type']}")
            
            columns_sql = ",\n                ".join(column_definitions)

            # 실행할 SQL 쿼리 (테이블 생성)
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_sql}
            );
            """

            # SQL 쿼리 실행
            cur.execute(create_table_query)

            # 변경사항을 데이터베이스에 커밋(commit)
            conn.commit()

            logger.info(f"'{table_name}' 테이블이 성공적으로 생성되었습니다.")

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error("Error while creating PostgreSQL table", error)
            if conn:
                conn.rollback()
        finally:
            # 연결 종료
            if conn is not None:
                cur.close()
                conn.close()
                logger.info("PostgreSQL connection is closed.")

    def insert_agent_data(self, table_name: str, data: dict):
        """
        """
        conn = None
        cur = None

        # 컬럼 이름을 SQL 쿼리 형식으로 변환
        columns = ['session_id', 'query', 'refined_query', 'answer', 'created_at', 'updated_at']
        columns_sql = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        
        # SQL 쿼리 생성
        sql = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"
        try:
            logger.info("START - INSERT DATA")
            # 데이터베이스 연결
            conn = self._get_db_connection()
            cur = conn.cursor()

            # 데이터 정규화            
            new_row = [
                data.get("session_id"),
                data.get("query"),
                data.get("refined_query"),
                data.get("answer"),
                data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
                data.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                ]

            cur.execute(sql, new_row)
            conn.commit()
                    
        except Exception as error:
            logger.error(f"전체 처리 중 오류 발생: {error}")
            if conn:
                conn.rollback()
        finally:
            # 리소스 정리
            if cur:
                cur.close()
            if conn:
                conn.close()
                logger.info("데이터베이스 연결 종료")

    def select_all_data(self, table_name: str, limit: Optional[int] = 10, order_by: str = "id"):
        """지정된 테이블의 10게 데이터를 조회합니다."""
        conn = None
        try:
            logger.info("START - SELECT DATA")
            # DB 연결
            conn = self._get_db_connection()
            cur = conn.cursor()

            # SELECT 쿼리 생성
            if limit:
                query = f"SELECT id, session_id, query, refined_query, answer FROM {table_name} ORDER BY {order_by} LIMIT %s;"
                cur.execute(query, (limit,))
            else:
                query = f"SELECT id, session_id, query, refined_query, answer FROM {table_name} ORDER BY {order_by};"
                cur.execute(query)

            # 조회된 모든 데이터를 한 번에 가져오기
            results = cur.fetchall()
            return results

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error("오류 발생:", error)
            return []
        finally:
            if conn is not None:
                cur.close()
                conn.close()

    def delete_data_by_id(self, table_name: str, id_column: str, record_id: str):
        """특정 ID를 가진 레코드를 테이블에서 삭제합니다."""
        conn = None
        deleted_rows = 0
        try:
            logger.info("START - DELETE DATA")
            # DB 연결
            conn = self._get_db_connection()
            cur = conn.cursor()

            record_id = str(record_id)  # ⭐ 핵심

            # DELETE 쿼리 실행
            delete_query = f"DELETE FROM {table_name} WHERE {id_column} = %s;"
            cur.execute(delete_query, (record_id,))

            # 삭제된 행 수 확인
            deleted_rows = cur.rowcount

            # 변경사항 커밋
            conn.commit()

            if deleted_rows > 0:
                logger.info(f"{table_name} 테이블에서 {id_column} {record_id}인 레코드가 성공적으로 삭제되었습니다.")
            else:
                logger.warning(f"{table_name} 테이블에서 {id_column} {record_id}인 레코드를 찾을 수 없습니다.")

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error("오류 발생:", error)
            # 오류 발생 시 롤백
            if conn is not None:
                conn.rollback()
        finally:
            if conn is not None:
                cur.close()
                conn.close()
        
        return deleted_rows

pg = PostgresPipeline() 

if __name__ == "__main__":


    # Delete Table
    # pg.drop_table(table_name="")

    # Create Table
    agent_schema = [
        {'name': 'id', 'type': 'BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY'}, 
        {'name': 'session_id', 'type': 'VARCHAR(300) NOT NULL'}, 
        {'name': 'query', 'type': 'TEXT NOT NULL'}, 
        {'name': 'refined_query', 'type': 'TEXT NOT NULL'}, 
        {'name': 'answer', 'type': 'TEXT NOT NULL'}, 
        {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"},
        {"name": "updated_at", "type": "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"}
        ]
    pg.create_table(table_name="agent", columns_config=agent_schema)


    # Insert Test


    
    pass

