# MLP_team9 - IT 직군 자소서 첨삭 AI 서비스 (Backend)


**IT 직군 취업준비생의 자기소개서와 채용 공고(JD)를 비교 분석하여, 합격 가능성을 높여주는 맞춤형 피드백 및 첨삭 제공 서비스**의 백엔드 서버입니다.

> **중요:** 본 백엔드 서버는 고성능 GPU가 필요한 AI 모델(`Ko-Qwen2-7B`)을 구동하기 위해 **Google Colab (GPU)** 환경에서 실행되며, **ngrok**을 통해 외부(프론트엔드)와 통신합니다.

---

## 프로젝트 핵심 아키텍처 (Hybrid LLM)

본 서비스는 **정확한 분석**과 **자연스러운 문장 생성**을 위해 두 가지 모델을 결합한 하이브리드 방식을 사용합니다.

| 구분 | 모델명 | 구동 환경 | 역할 |
| :--- | :--- | :--- | :--- |
| **분석 모델** | **Ko-Qwen2-7B-Instruct + LoRA** | Local (Colab GPU) | 채용 공고와 자소서를 분석하여 **구조화된 피드백(JSON)** 생성 (점수, 강점, 약점 등) |
| **첨삭 모델** | **Google Gemini-2.5-Flash** | Cloud API | 분석 데이터를 바탕으로 자연스러운 문장으로 **자소서 재작성(Rewrite)** |

---

## 🛠️ 기술 스택 (Tech Stack)

* **Environment:** Google Colab (Linux, A100/T4 GPU 권장)
* **Language:** Python 3.10+
* **Framework:** Flask, Flask-CORS
* **Network Tunneling:** ngrok (pyngrok)
* **AI & ML Libraries:**
    * `torch`, `transformers`, `peft` (LoRA), `accelerate`
    * `google-generativeai` (Gemini SDK)

---

## 백엔드 실행 가이드 (Google Colab)

### 1. 사전 준비 (Google Drive)
로컬에서 작업한 백엔드 프로젝트 폴더 전체를 **Google Drive**에 업로드합니다.
* **필수 파일:** `app.py`, `run_model.py`, `requirements.txt`, `my_finetuned_model_v2/` (폴더)
* **권장 경로:** `/content/drive/MyDrive/[패키지명]/`

### 2. Colab 환경 설정
1.  Google Colab 접속 및 새 노트 생성 (또는 업로드한 파일 열기)
2.  **런타임 유형 변경:**
    * 상단 메뉴 `런타임` > `런타임 유형 변경`
    * **하드웨어 가속기:** `GPU` 선택 (**T4** 또는 **A100** 권장)

### 3. 실행 코드 (Colab 셀 입력)

#### Step 1: 구글 드라이브 마운트 및 경로 이동
```python
from google.colab import drive
import os

drive.mount('/content/drive')
# [수정 필요] 본인의 구글 드라이브에 업로드한 폴더명으로 변경하세요
os.chdir('/content/drive/MyDrive/backend_test_final') 
print("현재 경로:", os.getcwd())
```

#### Step 2: 필수 라이브러리 및 ngrok 설치 ####
```Python

!pip install -r requirements.txt
!pip install flask-cors pyngrok
```
#### Step 3: ngrok 인증 (최초 1회) ####
ngrok 대시보드에서 Authtoken을 복사하여 입력합니다.
```
Python

# [수정 필요] 복사한 토큰을 붙여넣으세요
!ngrok config add-authtoken [YOUR_NGROK_AUTH_TOKEN]
```
#### Step 4: 서버 실행 및 외부 URL 생성 ####
```
Python

from pyngrok import ngrok

# 1. 백그라운드에서 Flask 서버 실행 (nohup 사용)
# app.py가 있는 경로인지 확인 필수
get_ipython().system_raw('nohup python app.py &')

# 2. ngrok으로 포트 5000번 터널링 (Public URL 생성)
# 이미 열려있는 터널이 있다면 닫기
ngrok.kill()

# HTTP 터널 생성
public_url = ngrok.connect(5000).public_url

print("="*50)
print(f"백엔드 서버 외부 접속 URL: {public_url}")
print("="*50)
print("위 URL을 복사하여 프론트엔드 담당자에게 전달하세요.")
print("프론트엔드 .env 파일의 VITE_SERVER_URL_API 값을 이 URL로 설정해야 합니다.")
```
## API 명세 (API Specification) ##
자소서 분석 및 첨삭 요청
URL: [ngrok_public_url]/api/revise

Method: POST

Content-Type: application/json

Request Body
```
JSON

{
  "essay_text": "지원자의 자기소개서 내용...",
  "job_description": "지원하려는 채용 공고 내용..."
}
```
Response Body
```
JSON

{
  "feedback_result": "{ 'score': 85, 'strengths': [...], ... }",  // JSON 문자열 (로컬 모델 분석 결과)
  "feedback_rewrite": "수정 제안된 자소서 내용...",  // Text (Gemini 첨삭 결과)
  "cheer_message": "AI 분석과 첨삭이 완료되었습니다! 합격을 기원합니다."
}
```
## 주의 사항 (Troubleshooting) ##
Colab 세션 유지:

브라우저 탭을 닫거나 일정 시간 입력이 없으면 런타임이 초기화될 수 있습니다.

ngrok URL 변경:

Colab을 재실행할 때마다 ngrok URL이 새로 발급됩니다.

변경된 URL을 반드시 프론트엔드 .env 파일에 다시 업데이트해야 합니다.

보안 (Security):

.env 파일(API Key 포함)은 구글 드라이브 내 프로젝트 폴더에 위치해야 하며, 절대 GitHub에 업로드하지 마세요.
