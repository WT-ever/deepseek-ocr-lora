import os
import json
import signal
from PIL import Image
from tqdm import tqdm
import glob
import time
from contextlib import contextmanager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from unsloth import FastVisionModel
from transformers import AutoModel

BASE_MODEL_PATH = "./deepseek_ocr"
LORA_PATH = "./lora_model"
IMAGE_DIR = "/root/autodl-tmp/datasets/image/eval"
OUTPUT_PATH = "./result/eval"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "results.jsonl")
PROMPT = "<image>\ndocparse"

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
TIMEOUT_SECONDS = 120
SAVE_INTERVAL = 10


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")


@contextmanager
def time_limit(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    return None


def load_model():
    print("正在加载模型...")
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_PATH,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
    )

    model.load_adapter(LORA_PATH)
    FastVisionModel.for_inference(model)
    print("模型加载完成")
    return model, tokenizer


def get_image_files(image_dir):
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(image_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern_upper = os.path.join(image_dir, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern_upper))
    return sorted(image_files)


def process_single_image(model, tokenizer, image_path, temp_image_file="temp_inference.jpg"):
    start_time = time.time()
    try:
        with time_limit(TIMEOUT_SECONDS):
            image = Image.open(image_path).convert("RGB")
            image.save(temp_image_file)
            
            res, token_stats = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=temp_image_file,
                output_path=OUTPUT_PATH,
                image_size=640,
                base_size=1024,
                crop_mode=True,
                save_results=False,
                test_compress=False,
                eval_mode=True
            )
            
            elapsed_time = time.time() - start_time
            print(f"  处理耗时: {elapsed_time:.2f}秒")
            
            mem_info = get_memory_info()
            if mem_info:
                print(f"  显存: {mem_info['allocated']:.2f}GB / {mem_info['total']:.2f}GB ({mem_info['allocated']/mem_info['total']*100:.1f}%)")
            
            return res, token_stats
    except TimeoutException:
        elapsed_time = time.time() - start_time
        print(f"  处理超时 ({elapsed_time:.2f}秒)")
        return None, None
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  处理失败 ({elapsed_time:.2f}秒): {e}")
        import traceback
        traceback.print_exc()
        return None, None


def batch_inference(model, tokenizer, image_files, batch_size=1, output_file=OUTPUT_FILE):
    results = []
    temp_image_file = "temp_inference.jpg"
    failed_images = []
    
    print(f"\n开始批量推理，batch_size={batch_size}")
    print(f"总图片数: {len(image_files)}")
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="批量推理进度"):
        batch = image_files[i:i+batch_size]
        
        for image_path in batch:
            image_name = os.path.basename(image_path)
            print(f"\n处理图片 {i+1}/{len(image_files)}: {image_name}")
            
            suffix, token_stats = process_single_image(model, tokenizer, image_path, temp_image_file)
            
            if suffix is not None and token_stats is not None:
                result = {
                    "image": image_name,
                    "prefix": "<image>\\ndocparse",
                    "suffix": suffix,
                    "tokens": f"{token_stats['image_tokens']}:{token_stats['text_tokens']}:{token_stats['total_tokens']}"
                }
                results.append(result)
                print(f"✓ 已处理: {image_name}")
            else:
                print(f"✗ 处理失败: {image_name}")
                failed_images.append(image_name)
        
        torch.cuda.empty_cache()
        
        if (i // batch_size + 1) % SAVE_INTERVAL == 0:
            print(f"\n已处理 {len(results)} 张图片，保存中间结果...")
            save_results(results, output_file)
    
    if failed_images:
        print(f"\n失败的图片 ({len(failed_images)}张):")
        for img in failed_images[:10]:
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... 还有 {len(failed_images) - 10} 张")
    
    return results


def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n结果已保存到: {output_file}")
    print(f"总共处理: {len(results)} 张图片")


def main():
    print("="*60)
    print("DeepSeek OCR 批量推理脚本")
    print("="*60)
    
    model, tokenizer = load_model()
    
    image_files = get_image_files(IMAGE_DIR)
    print(f"\n找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("错误: 没有找到图片文件!")
        return
    
    print(f"GPU信息: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # infer函数只支持单张图片，所以batch_size固定为1
    batch_size = 1
    print(f"使用batch_size={batch_size}")
    
    # 可以在这里设置处理所有图片或部分图片
    # image_files = image_files[:20]  # 只处理前20张
    print(f"本次处理全部 {len(image_files)} 张图片")
    
    results = batch_inference(model, tokenizer, image_files, batch_size=batch_size, output_file=OUTPUT_FILE)
    
    save_results(results, OUTPUT_FILE)
    
    print("\n批量推理完成!")
    print(f"成功处理: {len(results)} 张图片")
    print(f"失败: {len(image_files) - len(results)} 张图片")


if __name__ == "__main__":
    main()
