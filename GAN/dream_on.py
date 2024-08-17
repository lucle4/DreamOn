import os
import csv
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import style_transfer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
directory = os.getcwd()

classes = ('benign', 'malignant', 'normal')

# hyperparameters
img_size = 256
n_classes = len(classes)
color_channels = 1
n_epochs = 2500
batch_size = 10
latent_size = 400
filter_size_g = filter_size_d = 64
lr_g = lr_d = 0.0002
beta_1 = 0.9
beta_2 = 0.999

# input directories
img_dir_sono = os.path.join(directory, 'BUSI/split/train')
img_dir_masks = os.path.join(directory, 'BUSI/split/masks')
label_dir = os.path.join(directory, 'BUSI/split/labels_train.csv')

# image transformation
transform = transforms.Compose([
    transforms.transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])


def compute_cls_acc(predictLabel, target):
    return (((predictLabel.argmax(dim=1) == target.argmax(dim=1)) / current_batch_size) * 100).sum()


def average(list):
    return sum(list) / len(list)


class CustomDataset(Dataset):
    def __init__(self, img_dir_sono, img_dir_masks, label_dir, transform=None):
        self.img_dir_sono = img_dir_sono
        self.img_dir_masks = img_dir_masks
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

        image1_path = os.path.join(self.img_dir_sono, filename)
        image2_path = os.path.join(self.img_dir_masks, filename)

        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)

        if random.random() <= 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image


class ImageDirectoryDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [filename for filename in os.listdir(directory) if filename.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(image_path).convert("L")  # convert to grayscale
        if self.transform is not None:
            image = self.transform(image)
        return image


# create evaluation input for G
fixed_latent = torch.randn(10, latent_size, device=device).repeat(1, n_classes).view(n_classes * 10, latent_size)

fixed_labels = torch.zeros(n_classes * 10, n_classes, device=device)
for j in range(10):
    for i in range(n_classes):
        fixed_labels[j * n_classes + i][i] = 1

masks_dataset = ImageDirectoryDataset(img_dir_masks, transform=transform)
sono_dataset = ImageDirectoryDataset(img_dir_sono, transform=transform)

n_images = len(fixed_labels)
fixed_masks = torch.empty(n_images, color_channels, img_size, img_size, device=device)
masks_original = torch.empty(n_images, color_channels, img_size, img_size, device=device)

for i, label in enumerate(fixed_labels):
    class_index = torch.argmax(label).item()

    class_name = classes[class_index]
    class_images = [image for image in masks_dataset.images if class_name in image]

    subset_dataset_masks = ImageDirectoryDataset(img_dir_masks, transform=transform)
    subset_dataset_masks.images = class_images

    subset_dataset_sono = ImageDirectoryDataset(img_dir_sono, transform=transform)
    subset_dataset_sono.images = class_images

    random_image_index = random.randint(0, len(subset_dataset_masks) - 1)
    mask = subset_dataset_masks[random_image_index]
    sono = subset_dataset_sono[random_image_index]

    fixed_masks[i] = mask
    masks_original[i] = sono


class Generator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(Generator, self).__init__()

        # latent + label
        self.embedding = nn.Linear(n_classes, latent_size)

        self.layer1_latent = nn.Sequential(nn.ConvTranspose2d(latent_size, nb_filter * 2, 4, 1, 0, bias=False),
                                           nn.ReLU(True)
                                           )

        self.layer2_latent = nn.Sequential(nn.ConvTranspose2d(nb_filter * 2, nb_filter * 4, 4, 2, 1, bias=False),
                                           nn.BatchNorm2d(nb_filter * 4),
                                           nn.ReLU(True)
                                           )

        # mask
        self.layer1_mask = nn.Sequential(nn.Conv2d(color_channels, nb_filter, 3, 1, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        self.layer2_mask = nn.Sequential(nn.Conv2d(nb_filter, nb_filter * 2, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter * 2),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        self.layer3_mask = nn.Sequential(nn.Conv2d(nb_filter * 2, nb_filter * 4, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter * 4),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        self.layer4_mask = nn.Sequential(nn.Conv2d(nb_filter * 4, nb_filter * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter * 8),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        self.layer5_mask = nn.Sequential(nn.Conv2d(nb_filter * 8, nb_filter * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter * 8),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        self.layer6_mask = nn.Sequential(nn.Conv2d(nb_filter * 8, nb_filter * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(nb_filter * 8),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Dropout2d(0.5)
                                         )

        # latent + mask
        self.layer3_all = nn.Sequential(
            nn.ConvTranspose2d(nb_filter * 8 + nb_filter * 4, nb_filter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nb_filter * 8),
            nn.ReLU(True)
        )

        self.layer4_all = nn.Sequential(nn.ConvTranspose2d(nb_filter * 8, nb_filter * 8, 4, 2, 1, bias=False),
                                        nn.BatchNorm2d(nb_filter * 8),
                                        nn.ReLU(True)
                                        )

        self.layer5_all = nn.Sequential(nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1, bias=False),
                                        nn.BatchNorm2d(nb_filter * 4),
                                        nn.ReLU(True)
                                        )

        self.layer6_all = nn.Sequential(nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1, bias=False),
                                        nn.BatchNorm2d(nb_filter * 2),
                                        nn.ReLU(True)
                                        )

        self.layer7_all = nn.Sequential(nn.ConvTranspose2d(nb_filter * 2, nb_filter, 4, 2, 1, bias=False),
                                        nn.BatchNorm2d(nb_filter),
                                        nn.ReLU(True)
                                        )

        self.layer8_all = nn.Sequential(nn.ConvTranspose2d(nb_filter, color_channels, 3, 1, 1, bias=False),
                                        nn.Tanh()
                                        )

        self.__initialize_weights()

    def forward(self, latent, label, mask):
        label_embedding = self.embedding(label)
        x1 = torch.mul(label_embedding, latent)
        x1 = x1.view(x1.size(0), -1, 1, 1)
        x1 = self.layer1_latent(x1)
        x1 = self.layer2_latent(x1)

        x2 = self.layer1_mask(mask)
        x2 = self.layer2_mask(x2)
        x2 = self.layer3_mask(x2)
        xclone_3 = x2.clone()
        x2 = self.layer4_mask(x2)
        xclone_4 = x2.clone()
        x2 = self.layer5_mask(x2)
        xclone_5 = x2.clone()
        x2 = self.layer6_mask(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.layer3_all(x)
        x = x + xclone_5
        x = self.layer4_all(x)
        x = x + xclone_4
        x = self.layer5_all(x)
        x = x + xclone_3
        x = self.layer6_all(x)
        x = self.layer7_all(x)
        x = self.layer8_all(x)

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

        self.layer1 = nn.Sequential(nn.Conv2d(color_channels * 2, nb_filter, 4, 2, 1, bias=False),
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

    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
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


training_dataset = CustomDataset(img_dir_sono, img_dir_masks, label_dir, transform=transform)
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
G_adv_loss_epoch = []
G_aux_loss_epoch = []
G_style_loss_epoch = []
G_content_loss_epoch = []
D_loss_epoch = []
real_score_epoch = []
fake_score_epoch = []
real_cls_acc_epoch = []
fake_cls_acc_epoch = []

# training loop
for epoch in range(n_epochs):
    for i, (images, masks, target) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)
        target = target.to(device)

        current_batch_size = images.size()[0]

        realLabel = []
        fakeLabel = []

        for j in range(current_batch_size):
            real_smooth = round(random.uniform(0.95, 1.0), 2)
            realLabel.append(real_smooth)

            fake_smooth = round(random.uniform(0.0, 0.05), 2)
            fakeLabel.append(fake_smooth)

        realLabel = torch.FloatTensor(realLabel).to(device)
        fakeLabel = torch.FloatTensor(fakeLabel).to(device)

        ###########
        # TRAIN D #
        ###########

        # on real data
        predictR, predictRLabel = D(images, masks)

        loss_real_adv = criterion_adv(predictR, realLabel)
        loss_real_aux = criterion_aux(predictRLabel, target)

        # monitoring
        real_cls_acc = compute_cls_acc(predictRLabel, target)
        real_cls_acc_epoch.append(real_cls_acc)
        real_score = (predictR.sum() / current_batch_size)
        real_score_epoch.append(real_score)

        # on fake data
        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        fake_images = G(latent_value, target, masks)

        predictF, predictFLabel = D(fake_images, masks)

        loss_fake_adv = criterion_adv(predictF, fakeLabel)

        # monitoring
        fake_cls_acc = compute_cls_acc(predictFLabel, target)
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

        fake_images = G(latent_value, target, masks)

        predictF, predictFLabel = D(fake_images, masks)

        lossG_adv = criterion_adv(predictF, realLabel)
        lossG_aux = criterion_aux(predictFLabel, target)

        lossG_style, lossG_content = style_transfer.run_style_transfer(fake_images, images)

        lossG_style /= 100_000
        lossG_content /= 10

        lossG = lossG_adv + lossG_aux + lossG_style + lossG_content

        G_loss_epoch.append(lossG.item())
        G_adv_loss_epoch.append(lossG_adv.item())
        G_aux_loss_epoch.append(lossG_aux.item())
        G_style_loss_epoch.append(lossG_style)
        G_content_loss_epoch.append(lossG_content)

        optimizerD.zero_grad()
        optimizerG.zero_grad()

        lossG.backward()
        optimizerG.step()


    stats_epoch = {
        'epoch': epoch + 1,
        'G loss': f'{average(G_loss_epoch):.3f}',
        'G adv loss': f'{average(G_adv_loss_epoch):.3f}',
        'G aux loss': f'{average(G_aux_loss_epoch):.3f}',
        'G style loss': f'{average(G_style_loss_epoch):.3f}',
        'G content loss': f'{average(G_content_loss_epoch):.3f}',
        'D loss': f'{average(D_loss_epoch):.3f}',
        'fake score': f'{average(fake_score_epoch):.3f}',
        'real score': f'{average(real_score_epoch):.3f}',
        'real class acc.': f'{average(real_cls_acc_epoch):.1f}%',
        'fake class acc.': f'{average(fake_cls_acc_epoch):.1f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'G loss', 'G adv loss', 'G aux loss', 'G style loss', 'G content loss', 'D loss',
                  'fake score', 'real score', 'real class acc.', 'fake class acc.']

    with open('stats.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    G_loss_epoch = []
    G_adv_loss_epoch = []
    G_aux_loss_epoch = []
    G_style_loss_epoch = []
    G_content_loss_epoch = []
    D_loss_epoch = []
    real_score_epoch = []
    fake_score_epoch = []
    real_cls_acc_epoch = []
    fake_cls_acc_epoch = []

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = G(fixed_latent, fixed_labels, fixed_masks).detach().cpu()
            masks_original = masks_original.cpu()

            fake_original_combined = torch.cat((fake, masks_original), dim=1)
            fake_original_combined = fake_original_combined.view(30, 2, 1, 256, 256)
            fake_original_combined = fake_original_combined.permute(0, 2, 1, 3, 4)
            fake_original_combined = fake_original_combined.view(60, 1, 256, 256)

            transform_PIL = transforms.ToPILImage()
            img_list = [vutils.make_grid(torch.reshape(fake_original_combined, (n_classes * 20, color_channels,
                                                                                img_size, img_size)),
                                         nrow=n_classes * 2, normalize=True)]

            transform_PIL(img_list[-1]).save('epoch {}.png'.format(epoch + 1))

    if (epoch + 1) % 100 == 0 and (epoch + 1) >= 1000:
        torch.save(G.state_dict(), 'checkpoints_G/checkpoint epoch {}.pt'.format(epoch + 1))
        torch.save(D.state_dict(), 'checkpoints_D/checkpoint epoch {}.pt'.format(epoch + 1))

