import os
# 优先设置镜像站环境变量（必须放在所有 huggingface 相关 import 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"        # 强制 Hugging Face 进入离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制 Transformers 进入离线模式

# 从 Hugging Face 镜像站下载完整的DeepSeek OCR模型文件夹
from huggingface_hub import snapshot_download
snapshot_download("unsloth/DeepSeek-OCR", local_dir = "deepseek_ocr")

# 配置环境变量与模型列表
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from transformers import AutoModel
from datasets import load_dataset
import os
from datacollator import DeepSeekOCRDataCollator
from PIL import Image

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

# 使模型变为可训练的、带有 LoRA 适配器的模型
model = FastVisionModel.get_peft_model(
    model,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# instruction = "<image>\nFree OCR. "

def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "<|User|>",
                "content": "<image>\n" + sample["prefix"],
                "images": [os.path.join(IMAGE_FOLDER, sample["image"])],
            },
            {
                "role": "<|Assistant|>",
                "content": sample["suffix"],
            },
        ]
    }



# --- 配置本地路径 ---
JSONL_PATH = "./datasets/label/train.jsonl"  # 你的 jsonl 文件路径
IMAGE_FOLDER = "./datasets/image/train"    # 你的图片存放文件夹
# Load dataset
dataset = load_dataset("json", data_files=JSONL_PATH, split="train")

# 执行转换并过滤掉加载失败的数据
converted_dataset = dataset.map(convert_to_conversation)
converted_dataset = converted_dataset.filter(lambda x: x is not None)

# converted_dataset = [convert_to_conversation(sample) for sample in dataset]
converted_dataset[10] # 展示converted_dataset[10]


# 配置data collator和trainer
from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported
FastVisionModel.for_training(model) # Enable for training!
data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model = model,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    train_on_responses_only=True,
)
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator, # Must use!
    train_dataset = converted_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases
        dataloader_num_workers=2,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
    ),
)

# 进行微调
trainer_stats = trainer.train()


# 简单测试微调后的效果
# prompt = "<image>\nFree OCR. "
prompt = "docparse"
image_file = 'your_image.jpg'
output_path = 'result/finetune'

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file,
    output_path = output_path,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    save_results = True,
    test_compress = False)


# 保存Lora参数
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
# 如果改为True，即合并
if False: model.save_pretrained_merged("unsloth_finetune", tokenizer,)

# To export and save to your Hugging Face account
# if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token = "PUT_HERE")
