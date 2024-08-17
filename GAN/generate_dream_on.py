import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import random

device = 'cpu'

classes = ('benign', 'malignant', 'normal')

n_images = 600

w_true_benign = 0.56     # 337 images
w_true_malignant = 0.27  # 160 images
w_true_normal = 0.17     # 103 images

w_benign_lower = w_true_benign + 0.15
w_benign_upper = w_true_benign - 0.15
w_malignant_lower = w_true_malignant + 0.15
w_malignant_upper = w_true_malignant - 0.15
w_normal_lower = w_true_normal + 0.15
w_normal_upper = w_true_normal - 0.15

img_size = 256
n_classes = len(classes)
color_channels = 1
latent_size = 400
filter_size_g = 64
filter_size_g_mask = 96

classes_one_hot = F.one_hot(torch.arange(0, n_classes), num_classes=len(classes))


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


class MaskGenerator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(MaskGenerator, self).__init__()

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


G = Generator(latent_size, filter_size_g, n_classes).to(device)
G.load_state_dict(torch.load('checkpoint.pt', map_location=device))
G.eval()

G_Mask = MaskGenerator(latent_size, filter_size_g_mask, n_classes).to(device)
G_Mask.load_state_dict(torch.load('checkpoint_mask.pt', map_location=device))
G_Mask.eval()

transform_PIL = transforms.ToPILImage()


def weighing(label_one_hot, weight):
    label_one_hot = classes_one_hot[label_one_hot].to(device)
    label_one_hot = label_one_hot.tolist()
    label_one_hot = [item for elem in label_one_hot for item in elem]

    for i, value in enumerate(label_one_hot):
        if label_one_hot[i] == 1:
            label_one_hot[i] = weight

    label_one_hot = torch.FloatTensor(label_one_hot).to(device)

    return (label_one_hot)


w_current_benign = 0
w_current_malignant = 0
w_current_normal = 0

# generate the images
for i in range(n_images):
    rand_num = random.uniform(0, 1)

    if rand_num < w_true_benign:
        label_1 = torch.LongTensor([0])
        w_1 = random.uniform(w_benign_lower, w_benign_upper)
        w_2 = 1 - w_1

    elif rand_num < w_true_benign + w_true_malignant:
        label_1 = torch.LongTensor([1])
        w_1 = random.uniform(w_malignant_lower, w_malignant_upper)
        w_2 = 1 - w_1

    else:
        label_1 = torch.LongTensor([2])
        w_1 = random.uniform(w_normal_lower, w_normal_upper)
        w_2 = 1 - w_1

    rand_num = random.uniform(0, 1)

    if rand_num < w_true_benign:
        label_2 = torch.LongTensor([0])
    elif rand_num < w_true_benign + w_true_malignant:
        label_2 = torch.LongTensor([1])
    else:
        label_2 = torch.LongTensor([2])

    while label_1 == label_2:
        rand_num = random.uniform(0, 1)

        if rand_num < w_true_benign:
            label_2 = torch.LongTensor([0])
        elif rand_num < w_true_benign + w_true_malignant:
            label_2 = torch.LongTensor([1])
        else:
            label_2 = torch.LongTensor([2])

    if label_1 == 0:
        w_current_benign += w_1
    elif label_1 == 1:
        w_current_malignant += w_1
    elif label_1 == 2:
        w_current_normal += w_1

    if label_2 == 0:
        w_current_benign += w_2
    elif label_2 == 1:
        w_current_malignant += w_2
    elif label_2 == 2:
        w_current_normal += w_2

    label_1_one_hot = weighing(label_1, w_1).to(device)
    label_2_one_hot = weighing(label_2, w_2).to(device)

    label_combined = label_1_one_hot + label_2_one_hot.to(device)
    label_combined = label_combined.view(1, n_classes)

    latent_value = torch.randn(1, latent_size).to(device)

    mask = G_Mask(latent_value, label_combined)
    img = G(latent_value, label_combined, mask)

    save_image(img, f':/dream_on/{classes[label_2]}_{w_2:.2f}_{classes[label_1]}_{w_1:.2f}_{i}.png',
               padding=2, normalize=True)

    print(f'step {i + 1} done')

print('final weights:')
print(f'benign: {w_current_benign / n_images:.2f}')
print(f'malignant:{w_current_malignant / n_images:.2f}')
print(f'normal: {w_current_normal / n_images:.2f}')
print(f'deviations: '
      f'{w_current_benign / n_images - w_true_benign:.2f} / '
      f'{w_current_malignant / n_images - w_true_malignant:.2f} / '
      f'{w_current_normal / n_images - w_true_normal:.2f}')
