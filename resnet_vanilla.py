import os
import csv
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('benign', 'malignant', 'normal')

n_epochs = 50
batch_size = 20
lr = 0.001

img_size = 256
n_classes = len(classes)

directory = os.getcwd()

img_dir_original = os.path.join(directory, 'BUSI/split/train')
label_dir_original = os.path.join(directory, 'BUSI/split/labels_train.csv')

img_dir_test = os.path.join(directory, 'BUSI/split/test/original')
label_dir_test = os.path.join(directory, 'BUSI/split/labels_test.csv')


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


transform_train = transforms.Compose([
    transforms.transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = CustomDataset(img_dir_original, label_dir_original, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(img_dir_test, label_dir_test, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

stats = []
highest_test_accuracy = 0.0

for epoch in range(n_epochs):
    running_train_loss = 0.0
    running_train_accuracy = 0.0
    running_test_accuracy = 0.0
    total_train = 0
    total_test = 0

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        current_batch_size = images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        output = model(images.float())
        train_loss = criterion(output, labels)

        train_loss.backward()
        optimizer.step()

        _, labels = torch.max(labels, 1)
        _, predicted = torch.max(output, 1)

        running_train_loss += train_loss.item()
        running_train_accuracy += (predicted == labels).sum().item()

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            current_batch_size = images.size(0)

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            _, labels = torch.max(labels, 1)
            _, predicted = torch.max(output, 1)

            running_test_accuracy += (predicted == labels).sum().item()

    train_loss_epoch = running_train_loss / len(train_loader.sampler)
    train_accuracy = 100 * (running_train_accuracy / len(train_loader.sampler))
    test_accuracy = 100 * (running_test_accuracy / len(test_loader.sampler))

    stats_epoch = {
        'epoch': f'{epoch + 1}',
        'train loss': f'{train_loss_epoch:.3f}',
        'train accuracy': f'{train_accuracy:.2f}%',
        'test accuracy': f'{test_accuracy:.2f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'train loss', 'train accuracy', 'test accuracy']

    with open('stats_vanilla.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    if test_accuracy > highest_test_accuracy and epoch > 50:
        highest_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'checkpoints_vanilla/checkpoint epoch {}.pt'.format(epoch + 1))

    elif (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoints_vanilla/checkpoint epoch {}.pt'.format(epoch + 1))
