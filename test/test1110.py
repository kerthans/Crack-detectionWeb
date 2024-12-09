import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

# 重复定义 UNet 类
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DSConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias)
        # Pointwise convolution (1x1 convolution)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def compute_dropout_rate(num_neurons, base_rate=0.5, max_rate=0.7):
    # 基于神经元数量线性增加Dropout率
    return min(base_rate + (num_neurons / 1024) * (max_rate - base_rate), max_rate)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 定义Dropout率计算函数
        def dropout_layer(out_channels):
            rate = compute_dropout_rate(out_channels)
            return nn.Dropout2d(p=rate)

        # 编码器部分
        self.down1 = nn.Sequential(
            DSConv(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DSConv(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            dropout_layer(64),  # 根据神经元数量计算Dropout率
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down2 = nn.Sequential(
            DSConv(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DSConv(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            dropout_layer(128),  # 根据神经元数量计算Dropout率
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 中间层
        self.middle = nn.Sequential(
            DSConv(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DSConv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            dropout_layer(256),  # 根据神经元数量计算Dropout率
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 解码器部分
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DSConv(256, 128, kernel_size=3, padding=1),  # 注意这里输入通道数是256
            nn.ReLU(inplace=True),
            DSConv(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            dropout_layer(128)  # 根据神经元数量计算Dropout率
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DSConv(128, 64, kernel_size=3, padding=1),  # 注意这里输入通道数是128
            nn.ReLU(inplace=True),
            DSConv(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            dropout_layer(64)  # 根据神经元数量计算Dropout率
        )
        # 输出层
        self.outc = DSConv(64, 1, kernel_size=1)  # 最后一层输出1个通道
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样层，使输出与输入尺寸一致

    def forward(self, x):
        # 编码器
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        middle_out = self.middle(down2_out)

        # 解码器
        up1_out = self.up1(middle_out)
        up1_out = F.interpolate(up1_out, size=down2_out.size()[2:], mode='bilinear', align_corners=True)
        up1_out = torch.cat([up1_out, down2_out], dim=1)

        up2_out = self.up2(up1_out)
        up2_out = F.interpolate(up2_out, size=down1_out.size()[2:], mode='bilinear', align_corners=True)
        up2_out = torch.cat([up2_out, down1_out], dim=1)

        # 输出层
        out = self.outc(up2_out)

        return out

    def crop_tensors_to_match(self, tensor1, tensor2):
        _, _, height1, width1 = tensor1.size()
        _, _, height2, width2 = tensor2.size()

        if height1 > height2:
            diff_h = (height1 - height2) // 2
            tensor1 = tensor1[:, :, diff_h:(diff_h + height2), :]
        elif height1 < height2:
            diff_h = (height2 - height1) // 2
            tensor2 = tensor2[:, :, diff_h:(diff_h + height1), :]

        if width1 > width2:
            diff_w = (width1 - width2) // 2
            tensor1 = tensor1[:, :, :, diff_w:(diff_w + width2)]
        elif width1 < width2:
            diff_w = (width2 - width1) // 2
            tensor2 = tensor2[:, :, :, diff_w:(diff_w + width1)]

        return tensor1, tensor2

    def forward(self, x):
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        middle_out = self.middle(down2_out)
        up1_out = self.up1[0](middle_out)
        up1_out, down2_out = self.crop_tensors_to_match(up1_out, down2_out)
        up1_out = torch.cat([up1_out, down2_out], dim=1)
        up1_out = self.up1[1:](up1_out)  # 继续执行剩下的操作

        up2_out = self.up2[0](up1_out)
        up2_out, down1_out = self.crop_tensors_to_match(up2_out, down1_out)
        up2_out = torch.cat([up2_out, down1_out], dim=1)
        up2_out = self.up2[1:](up2_out)  # 继续执行剩下的操作

        out = self.outc(up2_out)
        out = self.upsample(out)  # 添加这一行以确保输出尺寸与输入尺寸一致
        return out

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
model_path = 'unet_model.onnx'

# 加载模型状态字典，并过滤掉额外的键
state_dict = torch.load(model_path, map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('.total_ops', '.total_params'))}

# 实例化模型
model = UNet().to(device)

# 加载过滤后的状态字典
model.load_state_dict(filtered_state_dict, strict=False)  # 使用strict=False忽略缺失的键
model.eval()  # 设置模型为评估模式

# 定义与训练时相同的数据变换
data_transforms = transforms.Compose([
    transforms.Resize((352, 352)),  # 调整为实际尺寸
    transforms.ToTensor(),
])

# 指定输入图片的文件夹路径
# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
project_dir = os.path.dirname(current_script_path)

# 使用相对路径定义数据集目录
input_folder = os.path.join(project_dir, 'test3')
output_folder = os.path.join(project_dir, 'results')


# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 对指定文件夹中的所有图片进行分割
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        image_path = os.path.join(input_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        # 应用与训练时相同的数据变换
        input_tensor = data_transforms(image).unsqueeze(0).to(device)  # 添加batch维度

        # 进行预测
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)  # 使用sigmoid激活函数得到概率图
            prediction = (output > 0.5).float()  # 二值化处理

        # 将预测结果转换回图像格式并保存
        prediction = prediction.squeeze().cpu().numpy() * 255  # 将[0,1]范围转换为[0,255]
        prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')  # 单通道灰度图
        prediction_image.save(os.path.join(output_folder, image_name))

print("Segmentation results saved to the 'results' folder.")