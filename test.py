#使用few_show_9模型对图片进行预测
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
import torch

# 将图片转化为Tensor
img = Image.open("horse4.png").convert("RGB")
img = transform(img)
img = img.unsqueeze(0)

# 加载模型
model = torch.load("few_show_9")
model.eval()

# 模型预测
output = model(img)
print(output)
print(output.argmax(1))
