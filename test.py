#使用few_show_9模型对图片进行预测
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
import torch

class FewShot(nn.Module):# 5层卷积，3层全连接

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding="same"),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#使用few_show_9模型对图片horse4.png进行预测
#加载模型
model = torch.load("few_show_9")
#加载图片
image_path = "./dog7.png"
image = Image.open(image_path)
#转换图片
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
image = transform(image)
# print(image.shape)
# print(image)
#扩展图片维度
image = torch.unsqueeze(image, dim=0)
# print(image.shape)
# print(image)
#预测
output = model(image)
# print(output)
# print(output.shape)
#获取预测结果
pred = torch.argmax(output, dim=1)
print(pred)