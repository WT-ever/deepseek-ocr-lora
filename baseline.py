from PIL import Image
# 从 Hugging Face 镜像站下载完整的DeepSeek OCR模型文件夹
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download("unsloth/DeepSeek-OCR", local_dir = "deepseek_ocr")

# 配置环境变量与模型列表
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from transformers import AutoModel
import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit", # Qwen 3 vision support
    "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# 加载模型
model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model = AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# 获取中文字符数据集
from datasets import load_dataset
# dataset = load_dataset("priyank-m/chinese_text_recognition", split = "train[:2000]")
# --- 配置本地路径 ---
JSONL_PATH = "./datasets/label/train.jsonl"  # 你的 jsonl 文件路径
IMAGE_FOLDER = "./datasets/image/train"    # 你的图片存放文件夹
# Load dataset
dataset = load_dataset("json", data_files=JSONL_PATH, split="train")

# 获取第 10 条数据作为测试
sample = dataset[10]
# 拼接完整的图片路径
img_path = os.path.join(IMAGE_FOLDER, sample['image'])

try:
    # 1. 打开图片
    image = Image.open(img_path).convert("RGB")
    # 2. 保存到本地供后续 infer 使用 (因为 infer 函数需要 image_file 路径)
    image_file = 'your_image.jpg'
    image.save(image_file)
    print(f"成功加载并保存图片: {img_path}")
except Exception as e:
    print(f"Error loading image {img_path}: {e}")
    exit(1) # 如果图片加载失败，直接退出

# --- 推理部分 ---
prompt = "<image>\nFree OCR. "
output_path = 'result/baseline'

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 调用推理
# 注意：image_file 传的是刚才保存的路径字符串 'your_image.jpg'
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=False
)

print("推理完成，结果已保存。")
