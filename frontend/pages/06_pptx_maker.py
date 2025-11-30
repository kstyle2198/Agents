import streamlit as st
import requests
import base64
import os
from typing import Optional

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Streamlit 앱 설정
st.set_page_config(
    page_title="PPTX Maker",
    page_icon="",
    layout="wide"
)

# 제목과 설명
st.title("PPTX Maker")
st.markdown("""
긴 텍스트를 입력하면 AI가 자동으로 요약하고 PowerPoint 프레젠테이션을 생성합니다.
""")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # API 엔드포인트 설정
    api_url = st.text_input(
        "API 엔드포인트 URL",
        value=f"{BASE_URL}/generate-pptx",
        help="PPTX 생성 API 엔드포인트 URL"
    )
    
    # 텍스트 분석 API 엔드포인트
    analyze_api_url = st.text_input(
        "텍스트 분석 API 엔드포인트 URL",
        value=f"{BASE_URL}/analyze-text",
        help="텍스트 분석 API 엔드포인트 URL"
    )
    
    st.markdown("---")
    st.markdown("### 사용 방법")
    st.markdown("""
    1. 텍스트를 입력란에 붙여넣기
    2. '텍스트 분석' 버튼으로 내용 미리보기
    3. 'PPTX 생성' 버튼 클릭
    4. 생성된 파일 다운로드
    """)

# 메인 컨텐츠 영역
def validate_text_input(text: str) -> tuple[bool, Optional[str]]:
    """텍스트 입력 유효성 검사"""
    if not text.strip():
        return False, "텍스트를 입력해주세요."
    if len(text.strip()) < 50:
        return False, "텍스트가 너무 짧습니다. 50자 이상 입력해주세요."
    return True, None

def download_pptx(file_content: bytes, filename: str = "summary_presentation.pptx"):
    """PPTX 파일 다운로드 처리"""
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64}" download="{filename}">📥 PPTX 파일 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)

def call_generate_pptx_api(api_url: str, text: str) -> tuple[bool, Optional[bytes], Optional[str]]:
    """PPTX 생성 API 호출"""
    try:
        response = requests.post(
            api_url,
            json={"text": text},
            timeout=300  # 5분 타임아웃
        )
        
        if response.status_code == 200:
            return True, response.content, None
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            return False, None, f"API 오류: {error_detail}"
            
    except requests.exceptions.RequestException as e:
        return False, None, f"API 연결 오류: {str(e)}"

def call_analyze_text_api(api_url: str, text: str) -> tuple[bool, Optional[dict], Optional[str]]:
    """텍스트 분석 API 호출"""
    try:
        response = requests.post(
            api_url,
            json={"text": text},
            timeout=60  # 1분 타임아웃
        )
        
        if response.status_code == 200:
            return True, response.json(), None
        else:
            error_detail = response.json().get("detail", "알 수 없는 오류")
            return False, None, f"API 오류: {error_detail}"
            
    except requests.exceptions.RequestException as e:
        return False, None, f"API 연결 오류: {str(e)}"

def display_analysis_results(analysis_data: dict):
    """텍스트 분석 결과 표시"""
    st.subheader("📋 텍스트 분석 결과")
    
    # 기본 정보
    col1, col2 = st.columns(2)
    with col1:
        st.metric("발견된 서브 주제 수", analysis_data.get("subtopics_count", 0))
    with col2:
        st.metric("상태", "분석 완료")
    
    # 서브 주제 상세 정보
    subtopics = analysis_data.get("subtopics", [])
    
    if not subtopics:
        st.warning("분석된 서브 주제가 없습니다.")
        return
    
    # 각 서브 주제별로 아코디언 생성
    for i, subtopic in enumerate(subtopics, 1):
        with st.expander(f"서브 주제 {i}: {subtopic.get('subtopic', '제목 없음')}"):
            # 핵심 문장 표시
            key_points = subtopic.get('key_points', [])
            if key_points:
                st.write("**핵심 내용:**")
                for j, point in enumerate(key_points, 1):
                    st.write(f"{j}. {point}")
            else:
                st.info("이 서브 주제에 대한 핵심 내용이 없습니다.")

# 텍스트 입력 영역
st.subheader("텍스트 입력")
input_text = st.text_area(
    "PPT로 변환할 텍스트를 입력하세요:",
    height=300,
    placeholder="여기에 긴 텍스트를 붙여넣으세요... (최소 50자 이상)",
    help="뉴스 기사, 보고서, 논문 등 긴 텍스트를 입력하면 AI가 자동으로 요약하여 PPT를 생성합니다."
)

# 텍스트 통계 표시
if input_text:
    text_length = len(input_text.strip())
    st.caption(f"입력된 텍스트 길이: {text_length}자")

# 버튼 영역
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    analyze_button = st.button(
        "🔍 텍스트 분석",
        use_container_width=True,
        disabled=not input_text.strip(),
        help="텍스트를 분석하여 서브 주제와 핵심 내용을 미리 확인합니다"
    )

with col2:
    generate_button = st.button(
        "🎯 PPTX 생성하기",
        type="primary",
        use_container_width=True,
        disabled=not input_text.strip(),
        help="분석된 내용을 바탕으로 PPTX 파일을 생성합니다"
    )

with col3:
    clear_button = st.button(
        "🗑️ 초기화",
        use_container_width=True,
        help="모든 입력과 결과를 초기화합니다"
    )

# 초기화 버튼 처리
if clear_button:
    st.rerun()

# 텍스트 분석 처리
if analyze_button:
    # 입력 검증
    is_valid, error_message = validate_text_input(input_text)
    
    if not is_valid:
        st.error(error_message)
    else:
        # 진행 상황 표시
        with st.spinner("텍스트를 분석 중입니다..."):
            # API 호출
            success, analysis_data, error_message = call_analyze_text_api(analyze_api_url, input_text)
        
        if success and analysis_data:
            st.success("✅ 텍스트 분석이 완료되었습니다!")
            display_analysis_results(analysis_data)
            
            # 분석 결과를 세션 상태에 저장 (PPTX 생성 시 활용 가능)
            st.session_state.last_analysis = analysis_data
            st.session_state.analyzed_text = input_text
            
        else:
            st.error(f"❌ 텍스트 분석 실패: {error_message}")

# PPTX 생성 처리
if generate_button:
    # 입력 검증
    is_valid, error_message = validate_text_input(input_text)
    
    if not is_valid:
        st.error(error_message)
    else:
        # 진행 상황 표시
        with st.spinner("AI가 텍스트를 분석하고 PPT를 생성 중입니다... (몇 분 정도 소요될 수 있습니다)"):
            # API 호출
            success, file_content, error_message = call_generate_pptx_api(api_url, input_text)
        
        if success and file_content:
            st.success("✅ PPTX 생성이 완료되었습니다!")
            
            # 다운로드 섹션
            st.subheader("📥 생성된 PPTX 다운로드")
            download_pptx(file_content)
            
            # 파일 정보 표시
            file_size = len(file_content) / 1024  # KB 단위
            st.info(f"파일 크기: {file_size:.1f} KB")
            
            # 미리보기 정보
            with st.expander("생성된 프레젠테이션 정보"):
                st.markdown("""
                - AI가 입력된 텍스트를 분석하여 주요 내용을 추출했습니다
                - 자동으로 구조화된 슬라이드 형식으로 구성되었습니다
                - 각 슬라이드는 논리적인 흐름에 따라 배열되었습니다
                """)
                
        else:
            st.error(f"❌ PPTX 생성 실패: {error_message}")

# 이전 분석 결과가 있으면 표시
if hasattr(st.session_state, 'last_analysis') and st.session_state.get('analyzed_text') == input_text:
    st.markdown("---")
    display_analysis_results(st.session_state.last_analysis)

# 푸터
st.markdown("---")
st.caption("© 2024 PPTX 생성기 - AI 기반 텍스트 요약 및 프레젠테이션 생성 도구")