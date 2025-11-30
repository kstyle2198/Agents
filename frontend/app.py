import os
import base64
import pathlib
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

from utils.style import HOVERING_EFFECT
# ==== Background Image ====
def get_base64_of_image(image_file):
    """이미지 파일을 Base64로 인코딩하여 문자열로 반환합니다."""
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file, overlay_color="rgba(255,255,255,0.5)"):
    """
    CSS를 사용하여 부드럽게 움직이는 배경 이미지와 오버레이를 설정합니다.
    """
    bin_str = get_base64_of_image(image_file)
    page_bg_img = f"""
    <style>
    /* 움직이는 애니메이션 효과 정의 */
    @keyframes panImage {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* 앱 전체 배경 설정 */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        position: relative;
        background: url("data:image/png;base64,{bin_str}") no-repeat center center fixed;
        /* 이미지를 화면보다 약간만 크게 만들어 자연스러운 움직임 유도 */
        background-size: 115% auto;
        /* ⭐️ 개선된 부분: 지속시간, 타이밍 함수, 반복 */
        animation: panImage 80s ease-in-out infinite;
    }}

    /* 배경 위 오버레이 효과 */
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stHeader"]::before {{
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background: {overlay_color};
        z-index: 0; /* 콘텐츠 뒤에 위치 */
    }}

    /* 콘텐츠가 오버레이 위에 오도록 설정 및 **글자색 검정으로 변경** */
    .stApp, [data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
        color: black; /* 기본 글자색을 검정으로 설정 (추가된 부분) */**
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 이미지 파일 경로 설정 (사용자 환경에 맞게 수정해주세요) ---
image_path = "./system_image/bg_img1.jpg"
if os.path.exists(image_path):
    # 오버레이 색상을 밝게 설정했으므로 글자색을 검정으로 변경하는 것이 가독성에 좋습니다.
    set_background(image_path, overlay_color="rgba(255,255,255,0.6)")
else:
    st.warning(f"배경 이미지 파일을 찾을 수 없습니다: {image_path}")

# Inject CSS style for Hover effect
st.markdown(HOVERING_EFFECT, unsafe_allow_html=True)

def make_hover_container(title:str, content:str, url:str, height:str = "auto"):
    st.markdown(f"""
            <a href="{url}" target="_blank" class="clickable-box-wrapper">
            <div class="hover-box" style="height: {height};">
                <h1>{title}</h1>
                <p>{content}</p></div>
            </a>
        """, unsafe_allow_html=True)
    
image_paths = [
    "./system_image/img1.jpg",
    "./system_image/img2.jpg",
    "./system_image/img3.jpg",
    "./system_image/img4.jpg",
]
# base64로 인코딩된 이미지 태그 생성 함수
def get_base64_img_tag(file_path):
    with open(file_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        return f'<img src="data:image/png;base64,{encoded}" style="width: 100%; position: absolute; opacity: 0; transition: opacity 1s;">'

# 이미지 태그 리스트 생성
image_tags = ''.join([get_base64_img_tag(path) for path in image_paths])

# HTML + JS 코드로 슬라이드쇼 구성
html_code = f"""
<div id="slideshow" style="position: relative; width: 100%; max-width: 800px; margin: auto; height: 500px;">
  {image_tags}
</div>

<script>
const slides = document.querySelectorAll("#slideshow img");
let current = 0;

function showNextSlide() {{
    slides[current].style.opacity = 0;
    current = (current + 1) % slides.length;
    slides[current].style.opacity = 1;
}}

slides[0].style.opacity = 1;
setInterval(showNextSlide, 3000);
</script>
"""

def make_home():
    with st.container():
        col11, col12, col13 = st.columns([2, 9, 2])
        with col11: pass
        with col12: 
            col111, col112, col113 = st.columns(3)

            with col111: 
                make_hover_container(title="Schedule Agent", content="Google Calendar Managing Agent", url="http://localhost:8501/Schedule_Agent", height="200px")
            with col112: 
                make_hover_container(title="MCP Agent", content="Based on MCP Tools(Web, Wiki, Arxiv)", url="http://localhost:8501/MCP_Agent", height="200px")
            with col113: 
                make_hover_container(title="SQL Agent", content="Based on Postgres RDB", url="http://localhost:8501/sql_agent", height="200px")
            
            col121, col122, col123 = st.columns(3)
            with col121: 
                make_hover_container(title="PPTX Maker", content="Convert Texts into PPTX Draft", url="http://localhost:8501/pptx_maker", height="200px")
            with col122: 
                components.html(html_code, height=300)
            with col123: 
                make_hover_container(title="Empty02", content="", url="", height="200px")

            col131, col132, col133 = st.columns(3)
            with col131: 
                make_hover_container(title="Rag Agent", content="Rag Agent based on ElasticSearch", url="http://localhost:8501/Rag_Agent", height="200px")
            with col132: 
                make_hover_container(title="Streaming Test", content="LangGraph Streaming Test", url="http://localhost:8501/stream_agent", height="200px")
            with col133: 
                make_hover_container(title="Empty04", content="", url="", height="200px")

        with col13: pass 

    st.markdown('<div class="st-emotion-cache-1vo6xi6">', unsafe_allow_html=True)

if __name__ == "__main__":
    
    with st.sidebar:
        st.info("[FastAPI Doc 이동하기](http://localhost:8000/docs)")

    make_home()

