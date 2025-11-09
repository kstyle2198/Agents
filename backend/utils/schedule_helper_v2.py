import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(override=True)
model_name = os.getenv("NO_THINK_MODEL")


# 로거 설정
import logging
logger = logging.getLogger("shcuedule_agent")

from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resouce_service,  # 라이브러리 3.0.0 버전 업 하면서 resource 오타 낸 듯
    )
from langchain_google_community._utils import (
    get_google_credentials,
    )


def make_api_resource(secret_filepath:str="./keys/secret.json", token_filepath:str="./keys/token.json"):
    try:
        credentials = get_google_credentials(
            token_file=token_filepath,
            scopes=["https://www.googleapis.com/auth/calendar"],
            client_secrets_file=secret_filepath,
        )
        api_resource = build_resouce_service(credentials=credentials)
        return api_resource
    except Exception as e:
        logger.error("ERROR in make_api_resource")

def add_meeting_event(calendar_id: str, start_datetime: datetime, summary: str = "회의", all_day: bool = None):
    """일정 추가 함수"""
    try:
        # all_day 값이 주어지지 않으면 start_datetime의 시간 여부로 자동 판정
        if all_day is None:
            all_day = (start_datetime.time() == datetime.min.time())  # 00:00:00이면 하루 종일로 간주
        logger.info(all_day)

        if all_day:
            logger.info("하루 종일 일정")
            event = {
                "summary": summary,
                "start": {
                    "date": start_datetime.date().isoformat(),
                    "timeZone": "Asia/Seoul",  # ✅ 타임존 추가
                },
                "end": {
                    "date": (start_datetime.date() + timedelta(days=1)).isoformat(),
                    "timeZone": "Asia/Seoul",  # ✅ 타임존 추가
                },
            }
        else:
            logger.info("시간 지정 일정")
            end_datetime = start_datetime + timedelta(hours=1)
            event = {
                "summary": summary,
                "start": {
                    "dateTime": start_datetime.isoformat(),
                    "timeZone": "Asia/Seoul",
                },
                "end": {
                    "dateTime": end_datetime.isoformat(),
                    "timeZone": "Asia/Seoul",
                },
            }
        api_resource = make_api_resource()
        created_event = api_resource.events().insert(calendarId=calendar_id, body=event).execute()
        logger.info(f"일정이 추가되었습니다: {created_event.get('htmlLink')}")
    except Exception as e:
        logger.error(f"ERROR in add_meeting_event - {e}")

def event_dict(calendar_id: str, date: datetime):
    """특정 날짜의 이벤트를 딕셔너리 형태로 반환"""
    time_min = date.isoformat() + "Z"
    time_max = (date + timedelta(days=1)).isoformat() + "Z"
    try:
        api_resource = make_api_resource()
        events_result = api_resource.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])
        results  = dict()
        for event in events:
            results[event.get('summary')] = event.get('id')

        return results
    except Exception as e:
        logger.error(f"ERROR in event_dict - {e}")

def update_event(calendar_id: str, event_id: str, start_datetime: datetime, summary: str = "회의", all_day: bool = None):
    """이벤트 업데이트 함수"""

    try:
        # 기존 이벤트 가져오기
        api_resource = make_api_resource()
        event = api_resource.events().get(calendarId=calendar_id, eventId=event_id).execute()

        # all_day 값 자동 판정 (입력 안하면 기존 설정 유지)
        if all_day is None:
            all_day = (start_datetime.time() == datetime.min.time())

        # 기존 event 정보 유지 + 변경 적용
        event["summary"] = summary

        if all_day:
            event["start"] = {
                "date": start_datetime.date().isoformat(),
                "timeZone": "Asia/Seoul",
            }
            event["end"] = {
                "date": (start_datetime.date() + timedelta(days=1)).isoformat(),
                "timeZone": "Asia/Seoul",
            }
        else:
            end_datetime = start_datetime + timedelta(hours=1)
            event["start"] = {
                "dateTime": start_datetime.isoformat(),
                "timeZone": "Asia/Seoul",
            }
            event["end"] = {
                "dateTime": end_datetime.isoformat(),
                "timeZone": "Asia/Seoul",
            }

        updated_event = api_resource.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event
        ).execute()

        logger.info(f"이벤트(ID: {event_id})가 업데이트되었습니다: {updated_event.get('htmlLink')}")

    except Exception as e:
        logger.error(f"이벤트(ID: {event_id}) 수정 중 오류 발생: {e}")

def delete_event(calendar_id: str, event_id: str):
    """이벤트 삭제 함수"""
    try:
        api_resource = make_api_resource()
        api_resource.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        logger.info(f"이벤트(ID: {event_id})가 삭제되었습니다.")
    except Exception as e:
        logger.error(f"이벤트 IF({event_id}) 삭제 중 오류 발생: {e}")

import os
import json
from datetime import datetime
from typing import List, Union
from typing import Optional
from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator

# 환경 변수 로드
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is missing")

# 출력 스키마 정의
class ScheduleInfo(BaseModel):
    job_type: str
    old_time: Optional[datetime] = None
    old_info: Optional[str] = None
    new_time: Optional[datetime] = None
    new_info: Optional[str] = None
    del_time: Optional[datetime] = None
    del_info: Optional[str] = None

# Groq 클라이언트 초기화
client = Groq(api_key=GROQ_API_KEY)

def get_event_id(user_input:str, event_dict: dict) -> str:
    """사용자 입력과 이벤트 딕셔너리를 비교하여 일치하는 이벤트 ID를 반환"""
    system_prompt = f"""
    You are an expert id finder.
    Blow Ref Events are consist of title and id (key, value) pair dictionary.
    Find the event that is as same as the input value among Ref Events presented and return the same item of that events.
    
    <Ref Events>
    {event_dict}

    <User Input>
    {user_input}

    Output MUST be valid JSON containing array of item
    """

    try:
        response = client.chat.completions.create(
            model=model_name,  # JSON 출력에 더 적합한 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # 약간의 창의성 허용
            max_tokens=100
        )
        json_str = response.choices[0].message.content        
        logger.info(json_str)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        # 단일 객체인 경우 리스트로 변환
        if isinstance(parsed, dict):
            parsed = [parsed]

        return parsed if parsed else None

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return None

def is_new_event(user_input: str, event_list: List[str]) -> bool:
    """사용자 입력이 기존 이벤트 목록에 없는 새로운 이벤트인지 확인"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # 기존 이벤트 리스트를 프롬프트에 포함
    event_list_text = "\n".join(f"- {event}" for event in event_list)
    
    system_prompt = f"""
    You are an expert calendar assistant. Today's date is {current_date} (Asia/Seoul timezone).
    You will determine if the user's input contains a new event **not already present** in the existing schedule.

    Type of Events : 미팅(meeting), 연차(full day off), 반차(half day off), 거점(remote office) 

    <Instructions>
    Compare the user's input to the following list of existing schedule items.
    If the user's input describes an event that already exists (even approximately), respond with:
        {{ "is_new": false }}
    If the user's input describes a completely new event (not in the list), respond with:
        {{ "is_new": true }}
    Return only the JSON object and nothing else.

    <Existing Events>
    {event_list_text}
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=100
        )
        json_str = response.choices[0].message.content
        logger.info(json_str)
        try:
            result = json.loads(json_str)
            return result.get("is_new", True)  # 기본값은 True (새 이벤트)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    except Exception as e:
        logger.error(f"Error during event check: {str(e)}")
        return True  # 에러 발생 시 기본적으로 새 이벤트라고 간주

def process_schedule_request(user_input: str) -> Union[List[ScheduleInfo], None]:
    """사용자 입력을 처리하여 일정 정보를 추출"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f">>>current_date: {current_date}")
    system_prompt = f"""
    You are an expert schedule assistant. 
    오늘은 {datetime.today()} 입니다 (Asia/Seoul timezone).
    시간이 특정되지 않은 경우는 00:00 처리해주세요.T
    If the year is not mentioned, just assume this year is {datetime.year}. (Asia/Seoul timezone).

    Type of Schedules : 미팅(meeting), 연차(full day off), 반차(half day off), 거점(remote office) 
    
    <Review instructions>
    1. define job type among New, Update, Delete based on the user input
    2. accroding to the job type, review step by step and extract below schedule information.

    - old_time : canceled schedule datetime  (only when job_type is Update)
    - old_info : canceled schedule information (only when job_type is Update)
    - new_time : new meeting or event datetime  (if not specified, maintain old_time)
    - new_info : new schedule information (if not specified, maintain old_info and Do not include time info)
    - del_time : meeting or event datetime to be removed 
    - del_info : meeting or event information to be removed  

    Do not include Text Info in datetime value.
    Generate final answer as a concise and compact 2~3 keywords in Korean Language
    Output MUST be valid JSON containing either a single object or array of objects."""

    try:
        response = client.chat.completions.create(
            model=model_name,  # JSON 출력에 더 적합한 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,  # 약간의 창의성 허용
            max_tokens=500
        )

        json_str = response.choices[0].message.content     
        logger.info(f">>>user_input: {user_input}")   
        logger.info(f">>>json_str: {json_str}")
        # JSON 파싱
        try:
            parsed = json.loads(json_str)
            logger.info(f">>>parsed: {parsed}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        # 단일 객체인 경우 리스트로 변환
        if isinstance(parsed, dict):
            parsed = [parsed]

        # 유효성 검사
        validated = []
        for idx, item in enumerate(parsed, 1):
            try:
                validated.append(ScheduleInfo(**item))
                logger.info(f"Item {idx} validated successfully")
            except ValidationError as ve:
                logger.error(f"Validation error in item {idx}: {ve}")
        return validated if validated else None

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return None
    
##---------------------------
def date_extracter(user_input:str) -> str:
    """사용자의 입력에서 날짜와 시간을 추출"""
    system_prompt = f"""
    당신의 역할은 사용자의 입력에서 날짜 또는 시간과 관련된 단어를 포착후 String 형식으로 추출하는 것입니다.
    오늘은 {datetime.today()} 입니다. (Asia/Seoul timezone).
    시간이 특정되지 않은 경우는 00:00 처리해주세요.
    
    <User Input>
    {user_input}
    </User Input>

    <Example>
    - user input: 10월 30일 오후 2시의 주요 일정을 브리핑
    - output : :2025-10-30, 14:00
    </Example>

    사용자의 입력에 응답하지 말고,
    입력에 포함된 날짜와 시간만 반환하고, 다른 부사적인 정보는 제외해주세요.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,  # JSON 출력에 더 적합한 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,  # 약간의 창의성 허용
            max_tokens=1000
        )
        res = response.choices[0].message.content        
        return res

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return None
    
def get_schedules(calendar_id: str, user_input:str):
    """특정 날짜의 이벤트를 딕셔너리 형태로 반환"""

    date_str = date_extracter(user_input=user_input)
    logger.info(f"Extacted Date: {date_str}")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        time_min = date.isoformat() + "Z"
        time_max = (date + timedelta(days=1)).isoformat() + "Z"
    except:
        date = datetime.strptime(date_str, "%Y-%m-%d, %H:%M")
        time_min = date.isoformat() + "Z"
        time_max = (date + timedelta(days=1)).isoformat() + "Z"
        
    api_resource = make_api_resource()
    events_result = api_resource.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    
    events = events_result.get("items", [])
    return events

def schedule_briefing(user_input:str, event_dict: dict) -> str:
    """사용자의 일정 브리핑"""
    system_prompt = f"""
    당신은 스마트하고 상냥한 일정관리 비서입니다.

    오늘은 {datetime.today()} 입니다. (Asia/Seoul timezone).
    아래 내용을 참고하여 일정과 시간을 분석후 브리핑해주세요.
    장단기 일정을 구분하여 브리핑 해주세요.

    <Ref Schedules>
    {event_dict}

    <User Input>
    {user_input}
    
    반드시 주어진 일정에 있는 내용만 브리핑하고 친절하고 정중한 ton & manner를 지켜주세요.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,  # JSON 출력에 더 적합한 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,  # 약간의 창의성 허용
            max_tokens=1000
        )
        res = response.choices[0].message.content            
        return res

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return None
    


