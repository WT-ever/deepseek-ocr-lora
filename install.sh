#!/bin/bash
pip install -r requirements.txt
# 单独安装 flash-attn 以确保参数生效
pip install flash-attn --no-build-isolation