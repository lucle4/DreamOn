import os
import csv
import pandas as pd
import torch
from torchvision import transforms
from torch.distributions.beta import Beta
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('benign', 'malignant', 'normal')

n_epochs = 300
batch_size = 32
alpha = 0.2
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999

img_size = 256
n_classes = len(classes)


directory = os.getcwd()

img_dir_original = ''
label_dir_original = ''

img_dir_test = ''
label_dir_test = ''


def mixup_data(images, labels, alpha):
    batch_size = images.size(0)
    weights = Beta(alpha, alpha).sample((batch_size,)).to(device)
    indices = torch.randperm(batch_size).to(device)

    mixed_images = weights.view(batch_size, 1, 1, 1) * images + (1 - weights.view(batch_size, 1, 1, 1)) * images[indices]
    mixed_labels = weights.view(batch_size, 1) * labels + (1 - weights.view(batch_size, 1)) * labels[indices]

    return mixed_images, mixed_labels


class CustomDataset(Dataset):
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


transform = transforms.Compose([
    transforms.transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])


train_dataset = CustomDataset(img_dir_original, label_dir_original, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(img_dir_test, label_dir_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr, betas=(beta_1, beta_2))

stats = []

for epoch in range(n_epochs):
    running_train_loss = 0.0
    running_test_loss = 0.0
    running_train_accuracy = 0.0
    running_test_accuracy = 0.0
    total_train = 0
    total_test = 0

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        current_batch_size = images.size()[0]

        images = images.to(device)
        labels = labels.to(device)

        mixed_images, mixed_labels = mixup_data(images, labels, alpha)
        print(mixed_labels)

        output = model(mixed_images.float())
        train_loss = criterion(output, mixed_labels)

        _, predicted = torch.max(output, 1)
        _, label = torch.max(mixed_labels, 1)

        train_loss.backward()
        optimizer.step()

        total_train += current_batch_size
        running_train_loss += train_loss.item()
        running_train_accuracy += (predicted == label).sum().item()

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            _, predicted = torch.max(output, 1)
            _, label = torch.max(labels, 1)

            total_test += current_batch_size
            running_test_loss += test_loss.item()
            running_test_accuracy += (predicted == label).sum().item()

    train_loss_epoch = running_train_loss / total_train
    test_loss_epoch = running_test_loss / total_test
    train_accuracy = 100 * running_train_accuracy / total_train
    test_accuracy = 100 * running_test_accuracy / total_test

    stats_epoch = {
        'epoch': f'{epoch}',
        'train loss': f'{train_loss_epoch:.3f}',
        'test loss': f'{test_loss_epoch:.3f}',
        'train accuracy': f'{train_accuracy:.2f}%',
        'test accuracy': f'{test_accuracy:.2f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'train loss', 'test loss', 'train accuracy', 'test accuracy']

    with open('stats_simple_mixup.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    torch.save(model.state_dict(), 'checkpoints_simple_mixup/checkpoint epoch {}.pt'.format(n_epochs))
