from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
lora_dir = "./experiments/test-emoti-real/emoti-lora-8b-v2"
output_dir = "./emoti-lora-ko-8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map="auto", 
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(base_model, lora_dir)
model = model.merge_and_unload()  # 💡 LoRA 병합
model.save_pretrained(output_dir)  # ✅ 병합된 모델 저장
tokenizer.save_pretrained(output_dir)
