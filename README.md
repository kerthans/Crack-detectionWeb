# 裂缝检测系统

基于深度学习的裂缝检测系统，使用 Flask + React 构建的 Web 应用，支持图像上传、自动分割、数据分析和历史记录管理。

## 功能特点

- 🖼️ 图像上传和实时预览
- 🔍 基于 ONNX 模型的裂缝检测
- 📊 自动计算裂缝长度、宽度和面积
- 📈 数据可视化和趋势分析
- 📝 历史记录管理和删除
- 🔄 自动监控新图像处理
- 📱 响应式设计，支持移动端

## 系统要求

- Python 3.7+
- Flask
- ONNX Runtime
- PyTorch
- Pillow
- NumPy
- scikit-image

## 文件结构

```
app/
├── app.py              # Flask 主应用
├── image/             # 临时图像存储
├── model/             # 模型文件
│   └── unet_model.onnx
├── static/            # 静态资源
│   └── index.html
├── history/           # 历史记录
│   ├── data/         # JSON 数据文件
│   ├── original/     # 原始图像
│   └── processed/    # 处理后图像
├── todo/             # 自动处理队列
└── logs/             # 日志文件
```

## 安装步骤

1. 创建并激活虚拟环境（可选）：
```bash
python3 -m venv .venv
压缩包内有对应的.venv虚拟环境，直接启动
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
压缩包内有对应的.venv虚拟环境，直接启动
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 创建必要的目录结构：
```bash
mkdir -p app/{image,model,static,history/{data,original,processed},todo,logs}
```

4. 将 ONNX 模型文件放入 model 目录：
```bash
cp your_model.onnx app/model/unet_model.onnx
```

## 运行应用

1. 启动 Flask 服务器：
```bash
python app.py
```

2. 在浏览器中访问：
```
http://localhost:5000
```

## 使用说明

### 网页上传图片

1. 点击"选择文件"按钮上传图片
2. 预览图片并确认
3. 点击"开始分割"进行处理
4. 查看分割结果和测量数据
5. 历史记录会自动保存

### 自动处理

1. 将需要处理的图片放入 `todo` 文件夹
2. 系统会每分钟自动检查并处理新图片
3. 处理结果会保存到历史记录中

### 历史记录管理

1. 右侧面板显示所有历史记录
2. 可以查看每条记录的详细信息
3. 点击删除按钮可以移除记录
4. 数据趋势图自动更新
5. 所有历史记录由`app/history/data/history.json`管理

## 注意事项

1. **文件格式**：支持常见图像格式（PNG, JPG, JPEG）
2. **图像尺寸**：建议使用 1024x1024 以下的图像
3. **存储空间**：定期清理历史记录和临时文件
4. **网络连接**：确保服务器正常运行

## 故障排除

1. 如果图片上传失败：
   - 检查图片格式和大小
   - 确认有写入权限

2. 如果分割结果不显示：
   - 检查浏览器控制台错误
   - 确认模型文件存在

3. 如果历史记录不更新：
   - 检查文件权限
   - 查看日志文件

## 开发说明

### 后端接口

- `GET /` - 主页
- `POST /api/segmentation` - 图像分割
- `GET /api/history` - 获取历史记录
- `DELETE /api/history/<timestamp>` - 删除记录

### 前端组件

- 实时图像预览
- 分割结果显示
- 测量数据展示
- 历史记录管理
- 数据趋势图表

## 维护建议

1. 定期检查日志文件
2. 清理历史记录和临时文件
3. 更新 ONNX 模型
4. 备份重要数据

## 贡献指南

欢迎提交问题和改进建议：

1. Fork 项目
2. 创建新分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License


## 更新日志

### v1.0.0 (2024-12-09)
- 初始版本发布
- 基本功能实现
- Web界面完成