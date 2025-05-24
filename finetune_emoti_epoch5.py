import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.basicConfig(level=logging.INFO)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F

# ✅ 모델과 출력 경로
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
output_dir = "./experiments/test-emoti-real/emoti-lora-8b-v3"  # (버전 구분)
dataset_name = "sseyeonn/emoti-chatbot-ko"
data_files = [f"d_cactus{i}_ko_final.jsonl" for i in range(1, 17)]  # 1~16

# ✅ 데이터셋 로드
dataset = load_dataset(dataset_name, data_files={"train": data_files}, split="train")

# ✅ 프롬프트 생성
def format_instruction(example):
    return {"text": example["dialogue"]}

dataset = dataset.map(format_instruction)

# ✅ 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ✅ pad_token 없으면 추가
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

def tokenize(example):
    output = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
    )
    output["labels"] = output["input_ids"].copy()
    return output

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# ✅ QLoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ✅ 토큰 추가 시 embedding resize 필요
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# ✅ LoRA 설정 (🔍 변경된 부분: target_modules → ["q_proj", "v_proj"])
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # ✅ 여기 변경
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ✅ TrainingArguments (🔍 learning_rate → 1e-4 로 변경)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=5,
    learning_rate=1e-4,  # ✅ 여기 변경
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=50,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=False,
    report_to="none",
)

# ✅ CustomTrainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# ✅ 체크포인트 이어받기
last_checkpoint = get_last_checkpoint(output_dir)
if last_checkpoint is not None:
    print(f"📦 체크포인트에서 재시작: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("📦 새로 시작합니다")
    trainer.train()

# ✅ 어댑터와 토크나이저 저장
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n🎉 파인튜닝 완료! LoRA adapter가 저장되었습니다:", output_dir)
