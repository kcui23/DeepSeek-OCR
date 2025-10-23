#!/bin/bash

# DeepSeek OCR Web App 启动脚本

echo "========================================="
echo "  DeepSeek OCR Web Interface"
echo "========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "main.py" ]; then
    echo "错误: 请在 web_app 目录下运行此脚本"
    exit 1
fi

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
pip show fastapi &> /dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖中..."
    pip install -r requirements.txt
fi

# 创建必要的目录
mkdir -p uploads outputs static

echo ""
echo "启动服务器..."
echo "访问地址: http://localhost:8000"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动服务
python main.py
