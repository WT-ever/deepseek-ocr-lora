#!/usr/bin/env python3

import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import torch
    from unsloth import FastVisionModel
    from transformers import AutoModel
    
    print("正在导入模型...")
    sys.path.insert(0, '/root/autodl-tmp/deepseek-ocr-lora')
    from deepseek_ocr.modeling_deepseekocr import decoder_token_count
    
    print(f"初始decoder_token_count: {decoder_token_count}")
    
    BASE_MODEL_PATH = "./deepseek_ocr"
    LORA_PATH = "./lora_model"
    IMAGE_FILE = "your_image.jpg"
    OUTPUT_PATH = "result/eval"
    PROMPT = "<image>\ndocparse"
    
    print("正在加载模型...")
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_PATH,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
    )
    
    print("正在加载LoRA...")
    model.load_adapter(LORA_PATH)
    
    print("切换到推理模式...")
    FastVisionModel.for_inference(model)
    
    print("开始推理...")
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
    
    print(f"\n推理结果: {res}")
    print("Inference done.")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需的依赖包")
except Exception as e:
    print(f"运行错误: {e}")
    import traceback
    traceback.print_exc()
