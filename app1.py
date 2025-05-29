import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import BaseModel
from typing import List

# FastAPI 앱 초기화
app = FastAPI()

# 모델 ID 및 QLoRA 설정
MODEL_ID = "sseyeonn/emoti-chatbot-lora-ko-8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},  # GPU 1번 지정
    low_cpu_mem_usage=True
)

model.eval()

# 입력 데이터 구조 정의
class ChatRequest(BaseModel):
    name: str
    gender: str
    issue: str
    counselor_type: str
    history: List[str]
    user_message: str

# 상담사 유형별 프롬프트 스타일 템플릿
def get_counselor_intro(name: str, issue: str, style: str) -> str:
    if style == "공감형":
        return f"안녕하세요 {name}님. '{issue}'에 대해 이야기해 주셔서 감사합니다. 저는 공감적으로 경청하며 함께 고민을 나눌게요."
    elif style == "조언형":
        return f"안녕하세요 {name}님. '{issue}' 문제, 같이 현실적으로 정리해봅시다. 제가 솔직하게 조언드릴게요."
    elif style == "유머러스형":
        return f"{name}님~ '{issue}'이요? 너무 심각하게만 생각 말아요! 우리 가볍게 풀면서 이야기해볼까요?"
    else:
        return f"안녕하세요 {name}님. '{issue}'에 대해 이야기해 주세요."

# 프롬프트 생성
def build_prompt(data: ChatRequest) -> str:
    intro = f"""상담 시작 전 정보:
이름: {data.name}
성별: {data.gender}
상담사 유형: {data.counselor_type}
고민: {data.issue}

상담사: {get_counselor_intro(data.name, data.issue, data.counselor_type)}
"""

    history_text = "\n".join(data.history)
    dialogue = f"{history_text}\n내담자: {data.user_message}\n상담사:"
    return f"{intro.strip()}\n\n{dialogue.strip()}"

# 후처리: 응답 정제
def clean_response(text: str) -> str:
    import re
    text = text.split("내담자:")[0]
    text = re.sub(r"[a-zA-Z]{4,}", "", text)
    text = re.sub(r"[~^=+#/\\:;\"\'|\[\](){}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(("다.", "요.", "죠.", "네요.")):
        text = re.sub(r"[^가-힣]+$", "", text).strip() + "."
    return text

# API 엔드포인트
@app.post("/generate")
async def generate(data: ChatRequest):
    try:
        prompt = build_prompt(data)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():  # ✅ 변경: 더 안전한 추론 모드
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                early_stopping=True,  # ✅ 추가: 너무 긴 출력 방지
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = decoded.split("상담사:")[-1]
        result = clean_response(result)

        return {"output": result}

    except Exception as e:  # ✅ 예외 처리 추가
        return {"error": f"모델 응답 생성 중 오류 발생: {str(e)}"}

# 실행 안내
#if __name__ == "__main__":
 #   import uvicorn
  #  print("✅ FastAPI 상담사 서버 실행 중: http://localhost:8010/docs")
   # uvicorn.run("app1:app", host="0.0.0.0", port=8010, reload=True)
