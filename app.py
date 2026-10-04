import streamlit as st
from langdetect import detect
from transformers import pipeline


KOREAN_MODEL_PATH = "./model/korean/news_topic_model"
ENGLISH_MODEL_PATH = "./model/english/news_topic_model"


@st.cache_resource
def load_models():

    korean_classifier = pipeline(
        "text-classification",
        model=KOREAN_MODEL_PATH,
        tokenizer=KOREAN_MODEL_PATH,
    )

    english_classifier = pipeline(
        "text-classification",
        model=ENGLISH_MODEL_PATH,
        tokenizer=ENGLISH_MODEL_PATH,
    )

    return korean_classifier, english_classifier


korean_classifier, english_classifier = load_models()


st.title("📰 다국어 뉴스 토픽 분류 서비스")

title = st.text_input(
    "뉴스 제목을 입력하세요."
)


if st.button("토픽 분류하기"):

    if not title.strip():

        st.warning("뉴스 제목을 입력해주세요.")

    else:

        # 언어 감지
        language = detect(title)

        st.write(
            f"감지된 언어: `{language}`"
        )


        # 한국어
        if language == "ko":

            result = korean_classifier(title)[0]

        # 영어
        elif language == "en":

            result = english_classifier(title)[0]

        # 지원하지 않는 언어
        else:

            st.error(
                "현재 한국어와 영어만 지원합니다."
            )

            st.stop()


        st.subheader("분류 결과")

        st.success(result["label"])

        st.write(
            f"신뢰도: {result['score']:.2%}"
        )