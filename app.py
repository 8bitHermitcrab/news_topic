import streamlit as st
from transformers import pipeline


# ========================================
# 1. 페이지 설정
# ========================================

st.set_page_config(
    page_title="뉴스 토픽 분류",
    page_icon="📰",
    layout="centered",
)


# ========================================
# 2. 모델 로드
# ========================================

MODEL_PATH = "./model/news_topic_model"


@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
    )

    return classifier


classifier = load_model()


# ========================================
# 3. UI
# ========================================

st.title("📰 뉴스 토픽 분류 서비스")

st.write(
    "뉴스 제목을 입력하면 AI가 뉴스의 토픽을 분류합니다."
)


# ========================================
# 4. 뉴스 제목 입력
# ========================================

title = st.text_input(
    "뉴스 제목",
    placeholder="예: 삼성전자가 새로운 반도체 기술을 공개했다."
)


# ========================================
# 5. 분류 버튼
# ========================================

if st.button("토픽 분류하기"):

    if not title.strip():

        st.warning(
            "뉴스 제목을 입력해주세요."
        )

    else:

        result = classifier(title)[0]

        category = result["label"]
        score = result["score"]


        # ========================================
        # 6. 결과 출력
        # ========================================

        st.subheader("분류 결과")

        st.success(category)

        st.write(
            f"신뢰도: **{score:.2%}**"
        )