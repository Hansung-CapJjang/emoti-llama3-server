import torch
import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Flask 앱 초기화
app = Flask(__name__)

# 모델 경로 (Hugging Face 업로드 모델 사용)
MODEL_ID = "sseyeonn/emoti-lora-ko-8b"

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 4bit QLoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 모델 로드 (merge된 모델)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# 응답 정제 함수
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

# POST /chat 엔드포인트
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("input", "")
    history = data.get("history", "")

    prompt = f"{history}\n내담자: {user_input}\n상담사:"

    inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)

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

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_raw = generated_text.split("상담사:")[-1]
    response_clean = clean_response(response_raw)

    return jsonify({"response": response_clean})


if __name__ == "__main__":
    print("✅ Flask 추론 서버 실행 중... http://localhost:8000/chat")
    app.run(host="0.0.0.0", port=8000)
