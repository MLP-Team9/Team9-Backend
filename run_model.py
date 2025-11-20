import os
# [설정] Windows 환경에서 OpenMP 라이브러리 충돌 방지 (Intel CPU 등에서 발생하는 에러 해결)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 환경 설정 및 API 키 로드
# -----------------------------------------------------------------------------
# .env 파일에서 환경변수 로드 (API 키 등 민감 정보 보안 유지)
load_dotenv()

# [외부 모델] Google Gemini API 설정
# 역할: 로컬 모델이 분석한 데이터를 바탕으로 자연스러운 문장으로 '자소서 첨삭(Rewrite)' 수행
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    external_model = genai.GenerativeModel('gemini-2.5-flash')
    print(f"API 연결 성공! (Gemini-2.5-flash 로드됨)")
else:
    print("API_KEY가 설정되지 않았습니다. 자소서 첨삭(Rewrite) 기능을 사용할 수 없습니다.")
    external_model = None


# -----------------------------------------------------------------------------
# 2. 로컬 AI 모델 (Ko-Qwen2 파인튜닝) 로드
# -----------------------------------------------------------------------------
print("=== 로컬 AI 모델 로드를 시작합니다... (서버 시작 시 1~2분 소요) ===")

# [모델 정보]
# Base Model: Qwen2-7B (한국어 성능이 우수한 오픈소스 모델)
# Adapter: LoRA (Low-Rank Adaptation) 방식으로 파인튜닝된 경량화 가중치
MODEL_ID = "spow12/Ko-Qwen2-7B-Instruct"
ADAPTER_PATH = "./my_finetuned_model_v2" 
MODEL_DTYPE = torch.bfloat16 # 메모리 효율과 성능을 위한 bfloat16 사용 (A100/T4 환경 권장)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # 1. 토크나이저 로드 (텍스트 <-> 벡터 변환)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # 2. 원본(베이스) 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE, # GPU 자동 할당
        torch_dtype=MODEL_DTYPE
    )

    # 3. 파인튜닝된 어댑터(LoRA) 결합
    # 설명: 전체 모델을 다시 학습하는 대신, 학습된 어댑터만 로드하여 효율적으로 동작
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() # 추론 모드 설정 (Dropout 등 비활성화)

    print(f"로컬 AI 모델 로드 완료 (Device: {DEVICE})")

except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("경로(ADAPTER_PATH)가 올바른지, GPU 메모리가 충분한지 확인해주세요.")
    raise e


# -----------------------------------------------------------------------------
# 3. 핵심 기능 함수 정의
# -----------------------------------------------------------------------------

def get_ai_feedback(job_desc: str, user_essay: str) -> str:
    """
    [1단계: 분석] 로컬 파인튜닝 모델(Ko-Qwen2)을 사용
    기능: 채용 공고와 자소서를 비교 분석하여 점수, 강점, 약점 등을 추출
    특징: Instruction Tuning을 통해 JSON 포맷으로만 출력하도록 훈련됨
    """
    
    # [프롬프트 엔지니어링] 모델에게 페르소나 부여 및 JSON 스키마 강제
    system_prompt = """당신은 IT 직군 전문 채용 어드바이저입니다. 채용 공고와 자기소개서를 비교 분석하여, 반드시 다음 JSON 스키마에 맞춰서만 응답해야 합니다.

{
  "score": "종합 점수(0-100 정수)",
  "strengths": [
    "자소서가 공고에 부합하는 강점 (문자열 배열)"
  ],
  "weaknesses": [
    "자소서가 공고에 비해 부족한 약점 (문자열 배열)"
  ],
  "missing_keywords": [
    "공고에는 있으나 자소서에 빠진 키워드 (문자열 배열)"
  ],
  "overall_advice": "종합적인 조언 (문자열)"
}
"""
    user_content = f"### 채용 공고:\n{job_desc}\n\n### 자기소개서:\n{user_essay}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    # Chat Template 적용 (모델이 학습된 대화 형식으로 변환)
    prompt_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )

    inputs = tokenizer(prompt_template, return_tensors="pt").to(DEVICE)

    # 모델 추론 (Generate)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024, # 출력 길이 제한
            do_sample=True,
            temperature=0.1, # [중요] 0.1로 낮게 설정하여 JSON 형식을 엄격하게 지키도록 유도 (창의성 억제)
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 결과 디코딩 (토큰 -> 텍스트)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output_text.strip()


def call_external_llm_for_rewrite(json_analysis: str, original_essay: str) -> str:
    """
    [2단계: 첨삭] 외부 생성형 AI (Gemini) 사용
    기능: 1단계의 분석 결과(JSON)를 참고하여 실제 자소서를 매끄럽게 수정(Rewrite)
    이유: 로컬 모델(7B)은 분석에 강하지만, 긴 문장 생성 능력은 거대 모델(Gemini)이 우수함 -> 하이브리드 접근
    """
    if not external_model:
        return "API 키가 설정되지 않아 첨삭 기능을 수행할 수 없습니다."

    # Gemini에게 부여할 프롬프트 (역할 및 Chain-of-Thought 유도)
    prompt = f"""
    당신은 IT 직군 전문 취업 컨설턴트입니다.
    아래 제공되는 [지원자의 자소서 초안]과 [AI 분석 결과]를 참고하여,
    지원자가 합격할 수 있도록 **자기소개서를 더 매끄럽고 전문적으로 수정(Rewrite)** 해주세요.

    ---
    [지원자 자소서 초안]:
    {original_essay}

    [AI 분석 결과 (참고용)]:
    {json_analysis}
    ---

    [작성 가이드라인]:
    1. 분석 결과의 '강점'은 더 부각시키고, '약점'이나 '누락된 키워드'는 자연스럽게 보완하여 작성해주세요.
    2. 글자 수는 원본과 비슷하게 유지하되, 문장력을 높이고 IT 전문 용어를 적절히 활용하세요.
    3. 서두와 결미를 포함한 완성된 하나의 글 형태로 출력해주세요.
    4. 별도의 인사말 없이 수정된 자소서 내용만 바로 출력하세요.
    """

    try:
        # Gemini API 호출
        response = external_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        error_msg = f"외부 AI 서비스 연결 중 오류 발생: {str(e)}"
        print(error_msg)
        return error_msg
