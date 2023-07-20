import os
import csv
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Beta
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('benign', 'malignant', 'normal')

n_epochs = 100
batch_size = 20
lr = 0.001
alpha = 0.4

img_size = 256
n_classes = len(classes)

directory = os.getcwd()

img_dir_original = os.path.join(directory, 'BUSI/split/train')
label_dir_original = os.path.join(directory, 'BUSI/split/labels_train.csv')

img_dir_evaluate = os.path.join(directory, 'BUSI/split/evaluate')
label_dir_evaluate = os.path.join(directory, 'BUSI/split/labels_evaluate.csv')

img_dir_test = os.path.join(directory, 'BUSI/split/test/original')
label_dir_test = os.path.join(directory, 'BUSI/split/labels_test.csv')


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

evaluate_dataset = CustomDataset(img_dir_evaluate, label_dir_evaluate, transform=transform)
evaluate_loader = DataLoader(evaluate_dataset, batch_size=1, shuffle=True)

test_dataset = CustomDataset(img_dir_test, label_dir_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))

stats = []
highest_evaluation_accuracy = 0.0


def overall_accuracy(targets, predictions):
    correct_predictions = 0
    total_predictions = 0

    for target, prediction in zip(targets, predictions):
        target_class = torch.argmax(target)
        prediction_class = torch.argmax(prediction)

        if target_class == prediction_class:
            correct_predictions += 1

        total_predictions += 1

    overall_acc = correct_predictions / total_predictions

    return overall_acc


def balanced_accuracy(targets, predictions):
    correct_per_class = [0] * len(classes)
    total_per_class = [0] * len(classes)

    for target, prediction in zip(targets, predictions):
        target_class = torch.argmax(target)
        prediction_class = torch.argmax(prediction)

        if target_class == prediction_class:
            correct_per_class[target_class] += 1

        total_per_class[target_class] += 1

    class_balanced_acc = [correct / total if total > 0 else 0 for correct, total in
                          zip(correct_per_class, total_per_class)]
    mean_balanced_acc = sum(class_balanced_acc) / len(classes)

    return mean_balanced_acc, class_balanced_acc


for epoch in range(n_epochs):
    running_train_loss = 0.0

    predictions_evaluate = []
    labels_evaluate = []

    predictions_test = []
    labels_test = []

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        current_batch_size = images.size()[0]

        images = images.to(device)
        labels = labels.to(device)

        mixed_images, mixed_labels = mixup_data(images, labels, alpha)

        output = model(mixed_images.float())

        train_loss = criterion(output, mixed_labels)

        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(evaluate_loader):
            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            predictions_evaluate.append(output)
            labels_evaluate.append(labels)

        for i, (images, labels) in enumerate(test_loader):
            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            predictions_test.append(output)
            labels_test.append(labels)

    train_loss_epoch = running_train_loss / len(train_loader)
    evaluate_balanced_accuracy, _ = balanced_accuracy(labels_evaluate, predictions_evaluate)
    evaluate_overall_accuracy = overall_accuracy(labels_evaluate, predictions_evaluate)
    test_balanced_accuracy, _ = balanced_accuracy(labels_test, predictions_test)
    test_overall_accuracy = overall_accuracy(labels_test, predictions_test)

    stats_epoch = {
        'epoch': f'{epoch + 1}',
        'train loss': f'{train_loss_epoch:.3f}',
        'evaluate balanced accuracy': f'{evaluate_balanced_accuracy * 100:.2f}%',
        'evaluate overall accuracy': f'{evaluate_overall_accuracy * 100:.2f}%',
        'test balanced accuracy': f'{test_balanced_accuracy * 100:.2f}%',
        'test overall accuracy': f'{test_overall_accuracy * 100:.2f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'train loss', 'evaluate balanced accuracy', 'evaluate overall accuracy',
                  'test balanced accuracy', 'test overall accuracy']

    with open('stats_simple_mixup.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    if evaluate_balanced_accuracy > highest_evaluation_accuracy:
        highest_evaluation_accuracy = evaluate_balanced_accuracy
        torch.save(model.state_dict(), 'checkpoints_simple_mixup/checkpoint highest accuracy.pt')

    elif (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoints_simple_mixup/checkpoint epoch {}.pt'.format(epoch + 1))
