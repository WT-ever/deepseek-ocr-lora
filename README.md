# 2026-大创--多模态文档解析
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
   # 1. 启动学术加速（例如）（可根据平台自己选择）
   source /etc/network_turbo
   
   # 2. 创建名为 ds_ocr 的新环境，并指定 python 版本
   conda create -n ds_ocr python=3.12 -y
   
   # 3. 激活环境
   conda activate ds_ocr
   
   # 4. 安装PyTorch2.6.0
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **其余库依赖**
   
   ```
   chmod +x ./install.sh    # 添加执行权限
   ./install.sh             # 执行安装脚本
   ```
   
2. 获取相关文件

   ```
   git clone https://github.com/WT-ever/deepseek-ocr-lora.git
   
   # 切换分支
   git checkout main
   
   # 之后将datasets目录放在同目录下
   ```
   
3. 下载基线模型并测试

   ```
   python baseline.py
   ```

4. 进行微调

   ```
   python finetune.py
   ```

5. 进行推理

   ```
   python eval.py
   ```

   