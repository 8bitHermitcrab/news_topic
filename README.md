# 설치 환경

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## venv 환경 작업 시작할 때
source .venv/bin/activate
## venv 환경 작업 끝낼 때
deactivate

## 버전 저장 방법
pip freeze > requirements.txt


# 실행 방법
streamlit run app.py

# 뉴스 토픽 분류 서비스
https://newstopic-8ht69orifvdkyrbe3ap4b5.streamlit.app/