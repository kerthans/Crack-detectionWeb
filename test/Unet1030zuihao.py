import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from thop import profile  # 用于计算FLOPs和参数量
import time  # 用于计算推理时间
import torch.nn.functional as F  # 导入F以便使用F.interpolate
import scipy.io as sio  # 用于保存.mat文件
import math

# 定义数据变换
data_transforms = transforms.Compose([
    transforms.Resize((352, 352)),  # 调整为实际尺寸
    transforms.ToTensor(),
])


# 自定义数据集类
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask.squeeze()


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
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样层，使输出与输入尺寸一致

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


# 新增函数：计算P、R、F1、IOU
def calculate_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum().item()
    union = (pred + target).gt(0.0).sum().item()
    iou = intersection / union if union > 0 else 0
    precision = intersection / pred.sum().item() if pred.sum().item() > 0 else 0
    recall = intersection / target.sum().item() if target.sum().item() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, iou


# ... [训练循环保持不变]

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
project_dir = os.path.dirname(current_script_path)

# 使用相对路径定义数据集目录
img_dir = os.path.join(project_dir, 'CamVid', 'images')
mask_dir = os.path.join(project_dir, 'CamVid', 'mask')

# 创建数据集
dataset = CrackDataset(img_dir, mask_dir, transform=data_transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 加载模型
model = UNet().to(device)

# 使用thop库计算FLOPs和参数量
input_tensor = torch.randn(1, 3, 352, 352).to(device)  # 假设输入为单个批次
flops, params = profile(model, inputs=(input_tensor,))
print(f"Model FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 收集训练过程中的损失
train_losses = []
val_losses = []

# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_train_loss = 0.0
    num_batches = 0

    # 训练阶段
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        # 前向传播
        outputs = model(inputs)

        # 确保标签尺寸与模型输出尺寸一致
        if outputs.shape[-2:] != labels.shape[-2:]:
            labels = F.interpolate(labels.float(), size=outputs.shape[-2:], mode='nearest').long()

        # 计算损失
        loss = criterion(outputs, labels.float())
        running_train_loss += loss.item()
        num_batches += 1

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存当前epoch的平均训练损失
    avg_train_loss = running_train_loss / num_batches
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        total_val_loss = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_iou = 0
        num_samples = 0
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)

            # 确保标签尺寸与模型输出尺寸一致
            if outputs.shape[-2:] != labels.shape[-2:]:
                labels = F.interpolate(labels.float(), size=outputs.shape[-2:], mode='nearest').long()

            loss = criterion(outputs, labels.float())
            total_val_loss += loss.item()

            # 计算每个批次的指标
            batch_precision, batch_recall, batch_f1, batch_iou = calculate_metrics(outputs, labels)
            total_precision += batch_precision
            total_recall += batch_recall
            total_f1 += batch_f1
            total_iou += batch_iou
            num_samples += 1

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_precision = total_precision / num_samples
        avg_recall = total_recall / num_samples
        avg_f1 = total_f1 / num_samples
        avg_iou = total_iou / num_samples

        # 保存当前epoch的平均验证损失
        val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, '
          f'F1: {avg_f1:.4f}, IOU: {avg_iou:.4f}')

# 在验证阶段之后，计算并打印指标
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    num_samples = 0
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        outputs = model(inputs)

        # 确保标签尺寸与模型输出尺寸一致
        if outputs.shape[-2:] != labels.shape[-2:]:
            labels = F.interpolate(labels.float(), size=outputs.shape[-2:], mode='nearest').long()

        loss = criterion(outputs, labels.float())
        total_loss += loss.item()

        # 计算每个批次的指标
        batch_precision, batch_recall, batch_f1, batch_iou = calculate_metrics(outputs, labels)
        total_precision += batch_precision
        total_recall += batch_recall
        total_f1 += batch_f1
        total_iou += batch_iou
        num_samples += 1

    avg_val_loss = total_loss / len(val_dataloader)
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples
    avg_iou = total_iou / num_samples

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, '
          f'F1: {avg_f1:.4f}, IOU: {avg_iou:.4f}')

# 测试推理时间
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 352, 352).to(device)
    # 预热模型
    for _ in range(10):
        _ = model(dummy_input)

    start_time = time.time()
    num_runs = 100  # 运行次数
    for _ in range(num_runs):
        _ = model(dummy_input)
    end_time = time.time()

    total_time = end_time - start_time
    avg_fps = num_runs / total_time
    print(f"Average Inference Time: {total_time / num_runs * 1000:.2f}ms, FPS: {avg_fps:.2f}")

# 保存模型
model_save_path = 'crack_segmentation_model109out2.pth'
torch.save(model.state_dict(), model_save_path)

# 保存所有需要的数据到.mat文件
data_to_save = {
    'model_path': model_save_path,
    'flops_G': flops / 1e9,
    'params_M': params / 1e6,
    'avg_fps': avg_fps,
    'avg_precision': avg_precision,
    'avg_recall': avg_recall,
    'avg_f1': avg_f1,
    'avg_iou': avg_iou,
    'train_losses': np.array(train_losses),
    'val_losses': np.array(val_losses)
}

# 替换字典键名中的点号
for key in list(data_to_save.keys()):
    if '.' in key:
        new_key = key.replace('.', '_')
        data_to_save[new_key] = data_to_save.pop(key)

# 保存数据到.mat文件
sio.savemat('model_metrics_and_performance109out2.mat', data_to_save)

print("Model and metrics saved successfully.")

# 将模型导出为ONNX格式
onnx_file_path = os.path.join(project_dir, 'unet_model.onnx')

# 导出模型
torch.onnx.export(
    model,                      # 要导出的模型
    dummy_input,                # 模型的虚拟输入
    onnx_file_path,             # 输出文件的路径
    export_params=True,         # 是否存储训练好的模型参数权重
    opset_version=11,           # ONNX 的 opset 版本
    do_constant_folding=True,   # 是否执行常量折叠优化
    input_names=['input'],      # 输入节点名称
    output_names=['output'],    # 输出节点名称
    dynamic_axes={'input': {0: 'batch_size'},  # 动态轴（可选），这里设置 batch_size 为动态
                  'output': {0: 'batch_size'}}
)

print(f"Model has been successfully exported to {onnx_file_path}")

