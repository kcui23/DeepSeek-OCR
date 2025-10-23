# DeepSeek OCR Web Interface

基于 FastAPI 和 vLLM 的 DeepSeek OCR 网页界面，提供类似 Gradio 的简洁交互体验。

## 功能特点

- 🖼️ **图片上传**: 支持拖拽或点击上传图片
- ✏️ **可编辑提示词**: 自定义 OCR 提示词
- ⚙️ **参数调节**: 实时调整图像处理参数
  - Base Size: 基础图像尺寸 (512-2048)
  - Image Size: 处理图像尺寸 (320-1280)
  - Crop Mode: 裁剪模式开关
  - Min/Max Crops: 最小/最大裁剪数量
- 📊 **多视图结果展示**:
  - 可视化结果：带边界框的标注图片
  - 处理结果：格式化后的文本结果
  - 原始结果：完整的原始输出
- 🎨 **现代简约设计**: 主题色 rgb(87, 107, 231)

## 项目结构

```
web_app/
├── main.py                 # FastAPI 主服务器
├── inference_engine.py     # vLLM 推理引擎封装
├── requirements.txt        # Python 依赖
├── static/                 # 前端静态文件
│   ├── index.html         # 主页面
│   ├── app.js            # 前端交互逻辑
│   └── styles.css        # 样式表
├── uploads/               # 上传图片存储目录
└── outputs/               # OCR 结果输出目录
```

## 安装依赖

```bash
# 安装 FastAPI 相关依赖
pip install -r requirements.txt

# 确保已安装 vLLM 和其他 DeepSeek OCR 依赖
# 参考上级目录的安装说明
```

## 启动服务

```bash
# 进入 web_app 目录
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/web_app

# 启动服务器
python main.py

# 或使用 uvicorn 启动
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务启动后，在浏览器访问: `http://localhost:8000`

## 使用方法

### 1. 上传图片
- 点击上传区域或拖拽图片到上传框
- 支持 JPG, PNG, BMP 等常见图片格式

### 2. 编辑提示词
默认提示词：
```
<image>Convert the following image to markdown format.
```

可根据需求自定义，例如：
- `<image>Extract all text from this image.`
- `<image>Identify and extract table data.`
- `<image>Convert this document to markdown with proper formatting.`

### 3. 调整参数
根据图片大小和复杂度调整参数：
- **Base Size**: 控制基础处理分辨率
- **Image Size**: 控制输入图像大小
- **Crop Mode**: 是否启用图像裁剪
- **Min/Max Crops**: 裁剪块的数量范围

### 4. 运行 OCR
点击"运行 OCR"按钮，等待处理完成

### 5. 查看结果
- **可视化结果**: 查看带有检测框的标注图片
- **处理结果**: 查看格式化的文本内容
- **原始结果**: 查看完整的原始输出

## API 端点

### POST /upload
上传图片文件

**请求**:
- `multipart/form-data`
- `file`: 图片文件

**响应**:
```json
{
  "success": true,
  "filename": "uuid.jpg",
  "path": "/path/to/uploaded/file"
}
```

### POST /inference
运行 OCR 推理

**请求**:
- `image_path`: 上传的图片路径
- `prompt`: 提示词
- `base_size`: 基础尺寸 (默认: 1024)
- `image_size`: 图像尺寸 (默认: 640)
- `crop_mode`: 裁剪模式 (默认: true)
- `min_crops`: 最小裁剪数 (默认: 2)
- `max_crops`: 最大裁剪数 (默认: 6)

**响应**:
```json
{
  "success": true,
  "output_id": "uuid",
  "raw_result": "原始结果文本",
  "processed_result": "处理后的结果文本",
  "has_visualized_image": true
}
```

### GET /output/{output_id}/{filename}
获取输出文件（图片、结果文件等）

### GET /health
健康检查

## 技术栈

- **后端**: FastAPI, vLLM, PyTorch
- **前端**: 原生 HTML/CSS/JavaScript
- **模型**: DeepSeek OCR
- **推理引擎**: vLLM AsyncLLMEngine

## 注意事项

1. 确保 GPU 可用且有足够显存
2. 首次启动会加载模型，需要一定时间
3. 建议使用现代浏览器（Chrome, Firefox, Safari, Edge）
4. 大图片处理时间较长，请耐心等待
5. 确保 `config.py` 中的 `MODEL_PATH` 配置正确

## 故障排除

### 模型加载失败
- 检查 `MODEL_PATH` 是否正确
- 确认模型文件完整
- 检查 GPU 显存是否充足

### 推理超时
- 尝试减小 `base_size` 和 `image_size`
- 降低 `max_crops` 参数
- 检查图片是否过大

### 页面无法访问
- 确认服务已正常启动
- 检查防火墙设置
- 尝试使用 `127.0.0.1:8000` 替代 `localhost:8000`

## 许可证

遵循 DeepSeek OCR 项目的许可证
