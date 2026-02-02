import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"        # 强制 Hugging Face 进入离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制 Transformers 进入离线模式

import torch
from unsloth import FastVisionModel
from transformers import AutoModel

# ========= 配置 =========
BASE_MODEL_PATH = "./deepseek_ocr"
LORA_PATH = "./lora_model"      # 你 finetune 保存的
IMAGE_FILE = "your_image.jpg"
OUTPUT_PATH = "result/eval"
PROMPT = "<image>\ndocparse"             # 和你训练时一致

# ========= 加载模型 =========
model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
)

# 加载 LoRA
model.load_adapter(LORA_PATH)

# 切换到推理模式（非常重要）
FastVisionModel.for_inference(model)

# ========= 推理 =========
res = model.infer(
    tokenizer,
    prompt=PROMPT,
    image_file=IMAGE_FILE,
    output_path=OUTPUT_PATH,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    save_results=True,
    test_compress=False,
)

print(f"res的值为{res}")
print("Inference done.")
