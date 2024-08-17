import numpy as np
import pandas as pd
import os
import csv
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
directory = os.getcwd()

classes = ('benign', 'malignant', 'normal')

img_size = 256
n_classes = len(classes)
color_channels = 1
n_epochs = 5000
batch_size = 64
latent_size = 400
filter_size_g = 96
filter_size_d = 64
lr_g = 0.0002
lr_d = 0.0002
beta_1 = 0.5
beta_2 = 0.999

# input directories
img_dir = os.path.join(directory, 'BUSI/masks')
label_dir = os.path.join(directory, 'BUSI/labels.csv')

# image transformation
transform = transforms.Compose([
    transforms.transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

# functions for monitoring
def compute_cls_acc(predictLabel, target):
    return (((predictLabel.argmax(dim=1) == target.argmax(dim=1)) / batch_size) * 100).sum()


def average(list):
    return sum(list) / len(list)


class Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.labels_df = pd.read_csv(label_dir, header=None)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        filename = self.labels_df.iloc[index, 0]
        label = self.labels_df.iloc[index, 1]

        label = [float(i) for i in label.split(',')]
        label = torch.tensor(label)

        image_path = os.path.join(self.img_dir, filename)

        image = self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image


fixed_latent = torch.randn(10, latent_size, device=device).repeat(1, n_classes).view(n_classes * 10, latent_size)
fixed_labels = torch.zeros(n_classes * 10, n_classes, device=device)

for j in range(10):
    for i in range(n_classes):
        fixed_labels[j * n_classes + i][i] = 1


class Generator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(Generator, self).__init__()

        self.embedding = nn.Linear(n_classes, latent_size)

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(latent_size, nb_filter * 32, 4, 1, 0, bias=False),
                                    nn.ReLU(True)
                                    )

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 32, nb_filter * 16, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 16),
                                    nn.ReLU(True)
                                    )

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 16, nb_filter * 8, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 8),
                                    nn.ReLU(True)
                                    )

        self.layer4 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 4),
                                    nn.ReLU(True)
                                    )

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 2),
                                    nn.ReLU(True)
                                    )

        self.layer6 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 2, nb_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter),
                                    nn.ReLU(True)
                                    )

        self.layer7 = nn.Sequential(nn.ConvTranspose2d(nb_filter, color_channels, 4, 2, 1, bias=False),
                                    nn.Tanh()
                                    )

        self.__initialize_weights()

    def forward(self, latent, label):
        label_embedding = self.embedding(label)
        x = torch.mul(label_embedding, latent)
        x = x.view(x.size(0), -1, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class Discriminator(nn.Module):

    def __init__(self, nb_filter, n_classes):
        super(Discriminator, self).__init__()
        self.nb_filter = nb_filter

        self.layer1 = nn.Sequential(nn.Conv2d(color_channels, nb_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(nb_filter, nb_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 2),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(nb_filter * 2, nb_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 4),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(nb_filter * 4, nb_filter * 8, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 8),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer5 = nn.Sequential(nn.Conv2d(nb_filter * 8, nb_filter * 16, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 16),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer6 = nn.Sequential(nn.Conv2d(nb_filter * 16, nb_filter * 32, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 32),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.adv = nn.Sequential(nn.Conv2d(nb_filter * 32, 1, 4, 1, 0, bias=False),
                                 nn.Sigmoid()
                                 )

        self.aux = nn.Sequential(nn.Conv2d(nb_filter * 32, n_classes, 4, 1, 0, bias=False),
                                 nn.LogSoftmax(dim=1)
                                 )

        self.__initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        adv = self.adv(x)
        aux = self.aux(x)

        adv = adv.view(-1)
        aux = aux.view(-1, n_classes)

        return adv, aux

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


training_dataset = Dataset(img_dir, label_dir, transform=transform)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

G = Generator(latent_size, filter_size_g, n_classes).to(device)
D = Discriminator(filter_size_d, n_classes).to(device)

optimizerG = torch.optim.Adam(G.parameters(), lr_g, betas=(beta_1, beta_2))
optimizerD = torch.optim.Adam(D.parameters(), lr_d, betas=(beta_1, beta_2))

# loss functions
criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss()

total_step = len(train_loader)

stats = []
G_loss_epoch = []
D_loss_epoch = []
real_score_epoch = []
fake_score_epoch = []
real_cls_acc_epoch = []
fake_cls_acc_epoch = []

for epoch in range(n_epochs):
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        current_batch_size = images.size()[0]

        realLabel = []
        fakeLabel = []

        for j in range(current_batch_size):
            real_smooth = round(random.uniform(0.9, 1.0), 2)
            realLabel.append(real_smooth)

            fake_smooth = round(random.uniform(0.0, 0.1), 2)
            fakeLabel.append(fake_smooth)

        realLabel = torch.FloatTensor(realLabel).to(device)
        fakeLabel = torch.FloatTensor(fakeLabel).to(device)

        ###########
        # TRAIN D #
        ###########

        # on real data
        predictR, predictRLabel = D(images.float())

        loss_real_adv = criterion_adv(predictR, realLabel)
        loss_real_aux = criterion_aux(predictRLabel, target)

        real_cls_acc = compute_cls_acc(predictRLabel, target)
        real_cls_acc_epoch.append(real_cls_acc)
        real_score = (predictR.sum() / current_batch_size)
        real_score_epoch.append(real_score)

        # on fake data
        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels_G = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), gen_labels_G] = 1.0

        fake_images = G(latent_value, cls_one_hot)

        predictF, predictFLabel = D(fake_images)

        loss_fake_adv = criterion_adv(predictF, fakeLabel)

        # monitoring
        fake_cls_acc = compute_cls_acc(predictFLabel, cls_one_hot)
        fake_cls_acc_epoch.append(fake_cls_acc)
        fake_score = (predictF.sum() / current_batch_size)
        fake_score_epoch.append(fake_score)

        lossD = loss_real_adv + loss_real_aux + loss_fake_adv
        D_loss_epoch.append(lossD.item())

        optimizerD.zero_grad()
        optimizerG.zero_grad()

        lossD.backward()
        optimizerD.step()

        ###########
        # TRAIN G #
        ###########

        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels_G = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), gen_labels_G] = 1.0

        fake_images = G(latent_value, cls_one_hot)

        predictF, predictFLabel = D(fake_images)

        lossG_adv = criterion_adv(predictF, realLabel)
        lossG_aux = criterion_aux(predictFLabel, cls_one_hot)

        lossG = lossG_adv + lossG_aux
        G_loss_epoch.append(lossG.item())

        optimizerD.zero_grad()
        optimizerG.zero_grad()

        lossG.backward()
        optimizerG.step()

        print('success')

    stats_epoch = {
        'epoch': epoch + 1,
        'G loss': f'{average(G_loss_epoch):.3f}',
        'D loss': f'{average(D_loss_epoch):.3f}',
        'fake score': f'{average(fake_score_epoch):.3f}',
        'real score': f'{average(real_score_epoch):.3f}',
        'real class acc.': f'{average(real_cls_acc_epoch):.1f}%',
        'fake class acc.': f'{average(fake_cls_acc_epoch):.1f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'G loss', 'D loss', 'fake score', 'real score', 'real class acc.', 'fake class acc.']

    with open('stats_mask_generation.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    G_loss_epoch = []
    D_loss_epoch = []
    real_score_epoch = []
    fake_score_epoch = []
    real_cls_acc_epoch = []
    fake_cls_acc_epoch = []

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = G(fixed_latent, fixed_labels).detach().cpu()
            transform_PIL = transforms.ToPILImage()
            img_list = [vutils.make_grid(torch.reshape(fake, (n_classes * 10, color_channels, img_size,
                                                              img_size)), nrow=n_classes, normalize=True)]

            transform_PIL(img_list[-1]).save('samples_mask_generation/epoch {}.png'.format(epoch + 1))

    if (epoch + 1) % 500 == 0:
        torch.save(G.state_dict(), 'checkpoints_mask_generation/checkpoint epoch {}.pt'.format(epoch + 1))
