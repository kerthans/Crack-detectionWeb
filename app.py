from flask import Flask, request, jsonify, send_file, send_from_directory
import base64
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms
import logging
import os
import json
import datetime
import shutil
from pathlib import Path
from skimage.morphology import skeletonize
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from colorama import init, Fore, Back, Style

# 初始化colorama用于彩色输出
init()

# 配置日志
def setup_logger():
    log_dir = Path('app/logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - '
               f'{Fore.BLUE}%(levelname)s{Style.RESET_ALL} - '
               f'{Fore.YELLOW}%(message)s{Style.RESET_ALL}',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# 初始化Flask应用
app = Flask(__name__, static_folder='static')

# 创建必要的目录结构
def check_directories():
    directories = [
        'app/todo',
        'app/history/data',
        'app/history/original',
        'app/history/processed',
        'app/logs'
    ]
    for directory in directories:
        if Path(directory).exists():
            logger.info(f'目录存在: {directory}')
        else:
            logger.error(f'目录不存在: {directory}')

# 加载ONNX模型
def load_model():
    try:
        model_path = Path('app/model/unet_model.onnx')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return ort.InferenceSession(str(model_path))
    except Exception as e:
        logger.error(f'{Fore.RED}模型加载失败: {str(e)}{Style.RESET_ALL}')
        raise

# 历史记录管理
class HistoryManager:
    def __init__(self):
        self.history_file = Path('app/history/data/history.json')
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.save_history([])

    def load_history(self):
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f'{Fore.RED}加载历史记录失败: {str(e)}{Style.RESET_ALL}')
            return []

    def save_history(self, history):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f'{Fore.RED}保存历史记录失败: {str(e)}{Style.RESET_ALL}')

    def add_record(self, record):
        history = self.load_history()
        history.append(record)
        self.save_history(history)

    def delete_record(self, timestamp):
        history = self.load_history()
        history = [h for h in history if h['timestamp'] != timestamp]
        self.save_history(history)

# 图像处理类
class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.transforms = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image):
        image = Image.fromarray(np.array(image)).convert("RGB")
        image = self.transforms(image)
        return image.unsqueeze(0).numpy()

    def postprocess_output(self, output):
        output = np.squeeze(output, axis=0)
        output = 1 / (1 + np.exp(-output))
        output = (output > 0.5).astype(np.uint8) * 255
        if len(output.shape) == 3 and output.shape[0] == 1:
            output = output[0]
        return Image.fromarray(output, mode='L')
    def create_overlay(self, original_image, segmented_image, color=(255, 0, 0), alpha=0.5):
        """
        创建一个带有彩色叠加层的图像
        
        Args:
            original_image (PIL.Image): 原始图像
            segmented_image (PIL.Image): 分割结果图像
            color (tuple): RGB颜色元组，默认为红色
            alpha (float): 透明度，0到1之间
        
        Returns:
            PIL.Image: 叠加后的图像
        """
        # 确保图像尺寸一致
        original_image = original_image.convert('RGBA')
        segmented_image = segmented_image.convert('L')
        
        if original_image.size != segmented_image.size:
            segmented_image = segmented_image.resize(original_image.size, Image.LANCZOS)
        
        # 创建彩色遮罩
        overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        for x in range(overlay.width):
            for y in range(overlay.height):
                if segmented_image.getpixel((x, y)) > 127:
                    overlay.putpixel((x, y), (*color, int(255 * alpha)))
        
        # 合并图层
        return Image.alpha_composite(original_image, overlay)
    def calculate_measurements(self, segmented_image):
        segmented_array = np.array(segmented_image)
        crack_pixels = np.sum(segmented_array > 127)
        pixel_size_cm = 0.01
        area_cm2 = crack_pixels * pixel_size_cm**2
        skeleton = skeletonize(segmented_array > 127)
        skeleton_pixels = np.sum(skeleton)
        length_cm = skeleton_pixels * pixel_size_cm
        width_cm = area_cm2 / length_cm if length_cm > 0 else 0
        return {
            'length': f"{length_cm:.2f}",
            'width': f"{width_cm:.2f}",
            'area': f"{area_cm2:.2f}"
        }

# 初始化全局对象
check_directories()
model = load_model()
image_processor = ImageProcessor(model)
history_manager = HistoryManager()

# 路由处理函数
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/api/download/<path:filename>')
def download_file(filename):
    try:
        file_path = Path(filename)
        if not str(file_path).startswith('app/history/'):
            raise ValueError("Invalid file path")
            
        directory = os.path.dirname(filename)
        file_name = os.path.basename(filename)
        
        logger.info(f'{Fore.GREEN}Downloading file: {filename}{Style.RESET_ALL}')
        return send_from_directory(directory, file_name, as_attachment=True)
    except Exception as e:
        logger.error(f'{Fore.RED}下载文件失败: {str(e)}{Style.RESET_ALL}')
        return jsonify({'success': False, 'error': str(e)}), 400
@app.route('/api/segmentation', methods=['POST'])
def segmentation():
    try:
        data = request.get_json()
        if 'image' not in data:
            raise ValueError("Missing 'image' field in request")

        # 解码并处理图像
        image_data = base64.b64decode(data['image'])
        original_image = Image.open(io.BytesIO(image_data))
        
        # 处理图像
        input_tensor = image_processor.preprocess_image(original_image)
        outputs = model.run(None, {'input': input_tensor})
        segmented_image = image_processor.postprocess_output(outputs[0])
        
        # 创建叠加图像
        overlay_image = image_processor.create_overlay(
            original_image.convert('RGBA'),
            segmented_image,
            color=(255, 0, 0),  # 红色
            alpha=0.5  # 50%透明度
        )
        
        # 保存结果
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存原始图像
        original_path = f'app/history/original/original_{timestamp}.png'
        original_image.save(original_path)
        
        # 保存处理后的图像
        processed_path = f'app/history/processed/segmented_{timestamp}.png'
        segmented_image.save(processed_path)
        
        # 保存叠加图像
        overlay_path = f'app/history/processed/overlay_{timestamp}.png'
        overlay_image.save(overlay_path)
        
        # 获取测量结果
        measurements = image_processor.calculate_measurements(segmented_image)
        
        # 创建记录
        record = {
            'timestamp': timestamp,
            'original_path': original_path,
            'processed_path': processed_path,
            'overlay_path': overlay_path,
            'download_original': f'/api/download/{original_path}',
            'download_processed': f'/api/download/{processed_path}',
            'download_overlay': f'/api/download/{overlay_path}',
            'measurements': measurements,
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加到历史记录
        history_manager.add_record(record)
        
        # 返回处理结果
        buffered = io.BytesIO()
        segmented_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 返回叠加图像
        overlay_buffered = io.BytesIO()
        overlay_image.save(overlay_buffered, format="PNG")
        overlay_str = base64.b64encode(overlay_buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'segmentedImage': img_str,
            'overlayImage': overlay_str,
            'record': record
        })

    except Exception as e:
        logger.error(f'{Fore.RED}处理失败: {str(e)}{Style.RESET_ALL}')
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        history = history_manager.load_history()
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/<timestamp>', methods=['DELETE'])
def delete_history(timestamp):
    try:
        history_manager.delete_record(timestamp)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_todo_images():
    todo_dir = Path('app/todo')
    try:
        for image_path in todo_dir.glob('*.png'):
            logger.info(f'{Fore.CYAN}Processing new image: {image_path}{Style.RESET_ALL}')
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                response = app.test_client().post('/api/segmentation', 
                    json={'image': image_data})
                if response.status_code == 200:
                    # 移动已处理的图像到历史目录
                    shutil.move(str(image_path), 
                              f'app/history/original/original_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    except Exception as e:
        logger.error(f'{Fore.RED}处理待办图像失败: {str(e)}{Style.RESET_ALL}')

# 设置定时任务
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=process_todo_images,
    #在这里调整间隔多少分钟来检查文件夹
    trigger=IntervalTrigger(minutes=0.1),
    id='process_todo_images',
    name='Process images in todo directory'
)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)