"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/4/8 9:14
@Software: PyCharm 
@File : DCGAN.py
"""


import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from tqdm import tqdm

import matplotlib.pyplot as plt
import imageio
import glob

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 128 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = 128 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# 预训练VAE 加速生成器的生成 让Decoder变成生成器
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.init_size = 128 // 4
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512 * self.init_size ** 2, latent_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.init_size = 128 // 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size ** 2),
            nn.ReLU(),
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x), self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.BCELoss(size_average=False)
        reconstruction_loss = BCE(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence



# 定义训练函数
def train(dataloader, discriminator, generator, optimizer_D, optimizer_G, criterion, num_epochs):
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(tqdm(dataloader)):
            # 将图像数据和标签移到设备上
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones((real_imgs.size(0), 1)).to(device)
            fake_labels = torch.zeros((real_imgs.size(0), 1)).to(device)
            # 训练判别器
            optimizer_D.zero_grad()
            # 训练判别器识别真实图像
            real_outputs = discriminator(real_imgs)
            loss_D_real = criterion(real_outputs, real_labels)
            # 训练判别器识别虚假图像
            z = torch.randn((real_imgs.size(0), latent_dim)).to(device)
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs.detach())
            loss_D_fake = criterion(fake_outputs, fake_labels)
            # 计算总损失并更新判别器参数
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            # 训练生成器
            optimizer_G.zero_grad()
            # 训练生成器生成虚假图像并让判别器认为是真实图像
            output = discriminator(fake_imgs)
            loss_G = criterion(output, real_labels)
            # 更新生成器参数
            loss_G.backward()
            optimizer_G.step()

        # 输出训练结果
        tqdm.write('Epoch [{}/{}], Step [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, len(dataloader), loss_D.item(), loss_G.item()))
        if epoch%10==0:
            save_image(fake_imgs[:25], f"../output/epoch_{epoch}.png", nrow=5, normalize=True)


        # 保存生成器模型
        torch.save(generator.state_dict(), '../resources/generator.pth')
        torch.save(discriminator.state_dict(), '../resources/discriminator.pth')


if __name__ == '__main__':
    # 定义超参数
    batch_size = 64
    latent_dim = 100

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    num_epochs = 10000

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    dataset = ImageFolder(root='../data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化生成器和判别器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # 载入预训练模型
    #generator.load_state_dict(torch.load('../resources/generator.pth'))
    #discriminator.load_state_dict(torch.load('../resources/discriminator.pth'))

    # 定义优化器和损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # 定义真实标签和虚假标签
    real_label = 1
    fake_label = 0

    # 定义损失函数
    criterion = nn.BCELoss()

    # 训练模型
    #train(dataloader, discriminator, generator, optimizer_D, optimizer_G, criterion, num_epochs)

    # 生成动态图片
    anim_file = './output/dcgan.gif'

    # 使用 glob 模块获取所有的 PNG 文件
    filenames = sorted(glob.glob('../output/epoch*.png'))

    # 使用 imageio.get_reader() 函数创建一个可迭代的图像读取器
    images = [imageio.imread(filename) for filename in filenames]
    fps = 5
    # 使用 imageio.mimsave() 函数将图像序列保存为 GIF 文件
    imageio.mimsave(anim_file, images, duration = 1000 / fps)