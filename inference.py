import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ✅ 모델 경로
base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
lora_repo = "./experiments/test-emoti-real/emoti-lora-8b-v2"

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ✅ QLoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ✅ base 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ LoRA 어댑터 로드 및 merge
model = PeftModel.from_pretrained(base_model, lora_repo)
model = model.merge_and_unload()  # LoRA adapter merge
model.eval()

# ✅ 개선된 prompt (메타정보 제거 + 대화 framing 문장 추가)
prompt = """다음은 상담사와 내담자 간의 실제 상담 대화입니다. 상담사는 따뜻하고 공감적으로 응답합니다.

상담사: 안녕하세요 조나단 씨, 오늘 기분은 어떠세요?
내담자: 네, 괜찮아요. 요즘 좀 우울한 기분이 드네요.
상담사:"""

# ✅ 토크나이징
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ✅ 텍스트 생성 (generation parameter 최적화)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,            # 더 긴 응답 허용
        do_sample=True,
        temperature=0.6,               # randomness 줄임
        top_p=0.9,
        top_k=40,
        repetition_penalty=2.0,        # 반복 억제 강화
        no_repeat_ngram_size=4,        # 4-gram 반복 억제
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# ✅ 결과 디코딩
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 추론 결과:\n")
print(generated)

# ✅ 상담사 응답 추출 (prompt 이후 텍스트 잘라서 추출)
if "상담사:" in generated:
    response = generated.split("상담사:")[-1].strip()
else:
    response = generated.strip()

# ✅ 결과 출력
if len(response) < 5:
    print("\n❌ 모델이 유의미한 응답을 생성하지 못했습니다.")
else:
    print("\n✅ 상담사 응답:")
    print(response)
