import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import BaseModel
from typing import List

# FastAPI 앱 초기화
app = FastAPI()

# 모델 ID 및 4bit 로딩 설정
MODEL_ID = "sseyeonn/emoti-lora-ko-8b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    name: str
    gender: str
    issue: str
    counselor_type: str  # 공감형, 조언형, 유머러스형
    history: List[str]   # ["내담자: ...", "상담사: ..."]
    user_message: str

# 스타일 프롬프트 문구
def get_style_instruction(style: str) -> str:
    if style == "공감형":
        return "당신은 공감적이고 따뜻한 말투로 대화하는 상담사입니다."
    elif style == "조언형":
        return "당신은 현실적이고 명확한 조언을 주는 상담사입니다."
    elif style == "유머러스형":
        return "당신은 유쾌하고 가볍게 분위기를 이끄는 상담사입니다."
    else:
        return "당신은 친절하고 경청하는 상담사입니다."

# 대화 시작 인사
def get_intro_sentence(name: str, issue: str, style: str) -> str:
    if style == "공감형":
        return f"안녕하세요 {name}님. '{issue}'에 대해 이야기해 주셔서 감사합니다. 함께 이야기 나누며 마음을 들여다봐요."
    elif style == "조언형":
        return f"{name}님, '{issue}' 문제에 대해 같이 현실적으로 정리해봅시다."
    elif style == "유머러스형":
        return f"{name}님~ '{issue}' 너무 심각하게만 생각 말아요! 가볍게 풀면서 이야기해봐요 :)"
    else:
        return f"{name}님, '{issue}'에 대해 이야기해 주세요."

# 프롬프트 생성
def build_prompt(data: ChatRequest) -> str:
    role = get_style_instruction(data.counselor_type)
    intro = get_intro_sentence(data.name, data.issue, data.counselor_type)

    dialogue = [f"상담사: {intro}"]
    dialogue += data.history  # history는 이미 '내담자:' '상담사:' 구조라고 가정
    dialogue.append(f"내담자: {data.user_message}")
    dialogue.append("상담사:")  # 모델이 이어서 말하도록 유도

    return f"{role}\n\n" + "\n".join(dialogue)

# 후처리 최소화
def clean_response(text: str) -> str:
    # 첫 번째 상담사 이후 텍스트 추출
    parts = text.split("상담사:")
    if len(parts) > 1:
        response = parts[-1]
    else:
        response = text

    # 내담자 이후 텍스트 제거
    response = response.split("내담자:")[0].strip()
    return response.strip()

# API 엔드포인트
@app.post("/generate")
async def generate(data: ChatRequest):
    prompt = build_prompt(data)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = clean_response(decoded)

    return {"output": result}

# 실행 안내
if __name__ == "__main__":
    import uvicorn
    print("✅ FastAPI 상담사 서버 실행 중: http://localhost:8000/docs")
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)
