# 2026-大创--多模态文档解析
2026大创--多模态文档解析

### 操作步骤

1. 环境配置

   **基础环境**
   
   | **组件**        | **官方推荐版本** |
   | --------------- | ---------------- |
   | **Python**      | **3.12.9**       |
   | **PyTorch**     | **2.6.0**        |
   | **CUDA**        | **11.8**         |
   
   ```
   # 先选择Miniconda，cuda11.8，进去后安装指令如下
   # 1. 创建名为 ds_ocr 的新环境，并指定 python 版本
   conda create -n ds_ocr python=3.12 -y
   
   # 2. 激活环境
   conda activate ds_ocr
   
   # 3. 安装PyTorch2.6.0
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **其余库依赖**
   
   ```
   pip install -r requirements.txt
   ```
   
2. Git 获取相关文件

   ```
   git clone https://github.com/WT-ever/deepseek-ocr-lora.git
   
   Username: WT-ever
   password: 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
   
   # 切换分支
   git checkout zh_test
   ```

3. 下载基线模型并测试

   ```
   python baseline.py
   ```

   