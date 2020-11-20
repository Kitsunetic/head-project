import os

import torch
import torch.nn as nn
import torch_burn as tb
import torchvision
from torchvision import transforms

latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])

ds = torchvision.datasets.MNIST(root='../../data/',
                                train=True,
                                transform=transform,
                                download=True)
# (1, 28, 28), 0~9

D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

D = D.cuda()
G = G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


class GANTrainer(tb.Trainer):
    def forward(self, data, is_train: bool):
        D, G = self.model
        optimD, optimG = self.optim

        image = data[0].flatten(1)
        batch_size = image.shape[0]
        real_label = data[1].reshape(batch_size, 1)  # 예제에서는 왜 real label을 안 썼지?
        fake_labels = torch.zeros(batch_size, 1)
        image, real_label, fake_labels = self.cuda(image, real_label, fake_labels)

        # Train D
        real_pred = D(image)
        d_loss_real = self.criterion(real_pred, real_label)

        z = torch.randn(batch_size, latent_size)
        z = self.cuda(z)
        fake_images = G(z)
        fake_score = D(fake_images)
        d_loss_fake = self.criterion(fake_score, fake_labels)

        # Backward D
        d_loss = d_loss_real + d_loss_fake
        optimD.zero_grad()
        d_loss.backward()
        optimD.step()

        # Train G
        z = torch.randn(batch_size, latent_size)
        z = self.cuda(z)
        fake_images = G(z)
        fake_labels = D(fake_images)
        g_loss = self.criterion(fake_labels, real_label)

        optimG.zero_grad()
        g_loss.backward()
        optimG.step()

        return image, real_pred, real_label, d_loss_real.item(), d_loss_fake.item(), g_loss.item()

    def backward(self, x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, is_train: bool):
        pass

    def loop(self, batch_idx, data, is_train: bool, losses: dict, logs: dict):
        x, pred, y, d_loss_real, d_loss_fake, g_loss = self.forward(data, is_train)

        name1 = 'd_loss_real'
        name2 = 'd_loss_fake'
        name3 = 'g_loss'
        if not is_train:
            name1 = 'val_' + name1
            name2 = 'val_' + name2
            name3 = 'val_' + name3

        losses[name1] = d_loss_real
        losses[name2] = d_loss_fake
        losses[name3] = g_loss
        logs[name1] = self._ignition_mean(logs[name1], d_loss_real, batch_idx)
        logs[name2] = self._ignition_mean(logs[name2], d_loss_fake, batch_idx)
        logs[name3] = self._ignition_mean(logs[name3], g_loss, batch_idx)

        return x, pred, y


trainer = GANTrainer((D, G), criterion, (d_optimizer, g_optimizer))
trainer.fit(ds, train_valid_split=0.2, num_epochs=num_epochs,
            batch_size=batch_size, pin_memory=True)
