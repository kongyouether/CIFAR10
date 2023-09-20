import torch.utils.data
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

data_train = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
data_test = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

data_train_size = len(data_train)
data_test_size = len(data_test)
print("------训练集大小{}------".format(data_train_size))
print("------测试集大小{}------".format(data_test_size))


dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=64)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=64)


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


few_shot = FewShot()


# 损失函数
loss_fun = nn.CrossEntropyLoss()


# 优化器
learning_rate = 1e-2
optimier = torch.optim.SGD(few_shot.parameters(), lr=learning_rate)

# 训练轮数
Epoch = 10

# 记录训练次数
train_step = 0
test_step = 0

writer = SummaryWriter("logs")

# 训练
for epoch in range(Epoch):
    print("------第{}轮训练开始------".format(epoch + 1))

    # 训练步骤
    few_shot.train()
    train_accuracy = 0
    for data in dataloader_train:
        imgs, label = data
        output = few_shot(imgs)
        loss = loss_fun(output, label)

        # 优化器优化模型
        optimier.zero_grad()
        loss.backward()
        optimier.step()

        train_step += 1

        accuracy = (output.argmax(1) == label).sum()
        train_accuracy += accuracy
        if train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), train_step)
            print("第{}次训练，LOSS值为：{}".format(train_step, loss.item()))

    # 测试
    few_shot.eval()
    loss_test = 0
    test_accuracy = 0
    with torch.no_grad():

        for data in dataloader_test:
            imgs, label = data
            output = few_shot(imgs)
            loss = loss_fun(output, label)
            loss_test += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            test_accuracy += accuracy

    test_step += 1
    print("第{}轮测试，LOSS值为：{}".format(epoch + 1, loss_test))
    writer.add_scalar("test_loss", loss_test, test_step)

    print("第{}轮训练，准确率为：{}".format(epoch + 1, train_accuracy / data_train_size))
    print("第{}轮测试，准确率为：{}".format(epoch + 1, test_accuracy / data_test_size))

    writer.add_scalar("train_accuracy", train_accuracy / data_train_size, test_step)
    writer.add_scalar("test_accuracy", test_accuracy / data_test_size, test_step)

    # 模型保存
    torch.save(few_shot, "few_show_{}".format(epoch))
    print("模型已保存")

writer.close()