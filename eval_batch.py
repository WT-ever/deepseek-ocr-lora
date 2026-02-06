import os
import sys
import json
import signal
from PIL import Image
from tqdm import tqdm
import glob
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
TIMEOUT_SECONDS = 120

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
            
            return res, token_stats
    except TimeoutException:
        print(f"处理图片 {os.path.basename(image_path)} 超时 (超过{TIMEOUT_SECONDS}秒)")
        return None, None
    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def batch_inference(model, tokenizer, image_files, batch_size=1, output_file=None, save_interval=10):
    results = []
    temp_image_file = "temp_inference.jpg"
    processed_count = 0
    total_saved = 0
    first_save = True
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="批量推理进度"):
        batch = image_files[i:i+batch_size]
        
        for image_path in batch:
            image_name = os.path.basename(image_path)
            
            suffix, token_stats = process_single_image(model, tokenizer, image_path, temp_image_file)
            
            if suffix is not None and token_stats is not None:
                result = {
                    "image": image_name,
                    "prefix": "<image>\\ndocparse",
                    "suffix": suffix,
                    "tokens": f"{token_stats['image_tokens']}:{token_stats['text_tokens']}:{token_stats['total_tokens']}"
                }
                results.append(result)
                processed_count += 1
                print(f"✓ 已处理: {image_name} (累计: {processed_count})")
            else:
                print(f"✗ 处理失败: {image_name}")
        
        torch.cuda.empty_cache()
        
        if output_file and len(results) >= save_interval:
            save_results(results[:save_interval], output_file, append=not first_save)
            total_saved += len(results[:save_interval])
            results = results[save_interval:]
            first_save = False
    
    if results:
        if output_file:
            save_results(results, output_file, append=not first_save)
            total_saved += len(results)
        return results, total_saved
    else:
        return [], total_saved


def save_results(results, output_file, append=False):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    if not append:
        print(f"\n结果已保存到: {output_file}")
    print(f"已保存 {len(results)} 条记录到: {output_file}")


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
    
    # 只处理前20张图片
    # image_files = image_files[:10]
    print(f"本次处理前 {len(image_files)} 张图片")
    
    results, total_saved = batch_inference(model, tokenizer, image_files, batch_size=1, output_file=OUTPUT_FILE, save_interval=10)
    
    print(f"\n批量推理完成!")
    print(f"总共成功处理并保存: {total_saved} 张图片")


if __name__ == "__main__":
    main()