import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 베이스 모델 경로 (LoRA 없이)
base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# QLoRA 설정 (4bit 양자화 사용)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 모델 로딩 (LoRA 없이)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# 응답 정리 함수
def clean_response(text: str) -> str:
    text = re.split(r"(내담자:|<br>|<hr>|[\*\-]{2,})", text)[0]
    text = re.sub(r"[a-zA-Z]{4,}", "", text)
    text = re.sub(r"[!?.]{2,}", ".", text)
    text = re.sub(r"[~^=+#/\\:;\"\'|\[\](){}]", "", text)
    text = re.sub(r"\d+[\/\d+]*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(("다.", "요.", "습니다.", "죠.", "네요.")):
        text = re.sub(r"[^.!?가-힣]+$", "", text).strip() + "."
    return text

# 역할 프롬프트 (대화 시작 시 한 번만 삽입)
role_prompt = (
    "당신은 공감적이고 따뜻한 말투로 사람의 고민을 진지하게 들어주는 전문 상담사입니다. "
    "상담사는 항상 부드럽고 자연스럽게 대화에 응답해야 합니다.\n\n"
)

# 대화 시작
print("베이스 모델 테스트 (LoRA 없이). '종료' 입력 시 종료됩니다.\n")
dialogue_history = role_prompt  # 처음에 역할 삽입

while True:
    user_input = input("내담자: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        break

    # 프롬프트에 대화 누적
    dialogue_history += f"내담자: {user_input}\n상담사:"

    # 모델 추론
    inputs = tokenizer(dialogue_history.strip(), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_raw = generated_text.split("상담사:")[-1]
    response_clean = clean_response(response_raw)

    if len(response_clean) < 5:
        print("\nX: 유의미한 상담사 응답을 생성하지 못했습니다.\n")
    else:
        print(f"\n상담사: {response_clean}\n")
        dialogue_history += f" {response_clean}\n"
