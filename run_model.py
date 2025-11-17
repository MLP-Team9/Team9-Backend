# run_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

print("=== AI 모델 로드를 시작합니다... (서버 시작 시 1~2분 소요될 수 있음) ===")

MODEL_ID = "spow12/Ko-Qwen2-7B-Instruct"
ADAPTER_PATH = "./my_finetuned_model" # 1단계에서 복사한 폴더 경로
MODEL_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# 16비트로 원본(베이스) 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=MODEL_DTYPE
)

# 훈련시킨 '어댑터' 로드 (파인튜닝 적용!)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # 추론 모드로 설정

print(f"=== AI 모델 로드 완료! (Device: {DEVICE}) ===")


def get_ai_feedback(job_desc: str, user_essay: str) -> str:
    """
    채용 공고와 자소서를 받아, 파인튜닝된 AI 모델의 피드백(JSON)을 반환합니다.
    """
    
    system_prompt = "당신은 IT 직군 전문 채용 어드바이저입니다. 채용 공고와 자기소개서를 비교 분석하여, 정해진 JSON 양식으로만 피드백을 제공해야 합니다."
    user_content = f"### 채용 공고:\n{job_desc}\n\n### 자기소개서:\n{user_essay}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    prompt_template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )

    inputs = tokenizer(prompt_template, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1, # [주석] 모델 창의성 0.1 지정 (JSON 형식을 정확히 지키기 위함)
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output_text.strip()