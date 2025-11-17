#  MLP_team9 - IT 직군 자소서 첨삭 서비스 (백엔드)

IT 직군 취업준비생의 자소서와 채용 공고를 분석하여 맞춤형 피드백을 제공하는 LLM 서비스의 백엔드 서버입니다.

## 1. 실행 환경
# python이 설치되어 있어야 함.
* Python 3.13 (자신의 파이썬 버전)
* Flask
* (기타 등등)

## 2. 실행 방법
# cmd 또는 powershell을 열고 입력합니다.
# 파일을 저장한 폴더의 경로로 이동합니다(cd 파일경로)
1.  가상환경 생성: `python -m venv venv`
2.  가상환경 활성화: `venv\Scripts\activate`
3.  필요 라이브러리 설치: `pip install -r requirements.txt`
4.  서버 실행: `python app.py`

# 서버 실행 확인 및 테스트
1. Running on http://127.0.0.1:5000 메시지가 뜨는지 확인합니다 -> app.py 코드가 5000번 코드에서 실행됨
2. 프론트엔드 부분은 본인의 react 프로젝트에서(localhost:3000)에서 http://localhost:5000/api/revise 주소로 api 테스트를 시작할 수 있습니다.

## 3. API 명세

* **POST** `/api/revise`
* **Request Body:**
    ```json
    {
      "essay_text": "...",
      "job_description": "..."
    }
    ```
* **Response Body:**
    ```json
    { "feedback_result": "..." }
    ```