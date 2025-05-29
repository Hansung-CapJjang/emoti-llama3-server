import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 모델 경로
base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
lora_repo = "./experiments/test-emoti-real/emoti-lora-8b-v3"

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 4bit QLoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 모델 로딩
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_repo)
model = model.merge_and_unload()
model.eval()

# 대화 이력 초기화
dialogue_history = ""

# 출력 필터링 함수
def clean_response(text: str) -> str:
    # 불필요한 문장 이후 자르기
    text = re.split(r"(내담자:|<br>|<hr>|[\*\-]{2,})", text)[0]

    # 영어 문장, 특수문자 반복 제거
    text = re.sub(r"[a-zA-Z]{4,}", "", text)
    text = re.sub(r"[!?.]{2,}", ".", text)                # !!! -> .
    text = re.sub(r"[~^=+#/\\:;\"\'|\[\](){}]", "", text)  # 기타 특수문자 제거
    text = re.sub(r"\d+[\/\d+]*", "", text)                # 1/2/3/4 제거
    text = re.sub(r"\s+", " ", text).strip()

    # 문장이 끝나지 않고 이어지는 경우 보완
    if not text.endswith(("다.", "요.", "습니다.", "죠.", "네요.")):
        text = re.sub(r"[^.!?가-힣]+$", "", text).strip() + "."

    return text


# 채팅 시작
print("상담 챗봇에 오신 것을 환영합니다. '종료'라고 입력하면 종료됩니다.\n")

while True:
    user_input = input("내담자: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        break

    # 프롬프트 구성 (대화체만)
    dialogue_history += f"내담자: {user_input}\n상담사:"

    # 모델 추론
    inputs = tokenizer(dialogue_history.strip(), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.6,
            no_repeat_ngram_size=6,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 응답 추출
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_raw = generated_text.split("상담사:")[-1]
    response_clean = clean_response(response_raw)

    if len(response_clean) < 5:
        print("\nX : 유의미한 상담사 응답을 생성하지 못했습니다.\n")
    else:
        print(f"\n상담사: {response_clean}\n")
        dialogue_history += f" {response_clean}\n"
