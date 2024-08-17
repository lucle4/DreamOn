import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

device = 'cpu'

classes = ('benign', 'malignant', 'normal')

n_images = 600
random_label = False
output_class = 'normal'

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


def weighing(label_one_hot):
    label_one_hot = classes_one_hot[label_one_hot].to(device)
    label_one_hot = label_one_hot.tolist()
    label_one_hot = [item for elem in label_one_hot for item in elem]

    label_one_hot = torch.FloatTensor(label_one_hot).to(device)

    return (label_one_hot)


# generate the images
for i in range(n_images):
    if random_label:
        label = torch.LongTensor(np.random.randint(0, n_classes, 1))

    else:
        label = classes.index(output_class)
        label = torch.LongTensor([label])

    label_one_hot = weighing(label).to(device)
    label_one_hot = label_one_hot.view(1, n_classes)

    latent_value = torch.randn(1, latent_size).to(device)

    mask = G_Mask(latent_value, label_one_hot)
    img = G(latent_value, label_one_hot, mask)

    save_image(img, './dream_off/{} {}.png'.format(classes[label], i + 1), padding=2, normalize=True)
