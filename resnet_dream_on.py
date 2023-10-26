import os
import csv
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('benign', 'malignant', 'normal')

n_epochs = 100
batch_size = 20
lr = 0.001

img_size = 256
n_classes = len(classes)

directory = os.getcwd()

img_dir_original = os.path.join(directory, 'BUSI/split/train')
label_dir_original = os.path.join(directory, 'BUSI/split/labels_train.csv')

img_dir_generated = os.path.join(directory, 'gen_dataset/w_interpolation')
label_dir_generated = os.path.join(directory, 'gen_dataset/w_interpolation_labels_as_file.csv')

img_dir_validate = os.path.join(directory, 'BUSI/split/validate')
label_dir_validate = os.path.join(directory, 'BUSI/split/labels_validate.csv')

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


transform = transforms.Compose([
    transforms.transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

original_dataset = CustomDataset(img_dir_original, label_dir_original, transform=transform)
generated_dataset = CustomDataset(img_dir_generated, label_dir_generated, transform=transform)

combined_dataset = ConcatDataset([original_dataset, generated_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = CustomDataset(img_dir_validate, label_dir_validate, transform=transform)
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

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

    predictions_validate = []
    labels_validate = []

    predictions_test = []
    labels_test = []

    for i, (images, labels) in enumerate(combined_loader):
        model.train()
        optimizer.zero_grad()

        current_batch_size = images.size()[0]

        images = images.to(device)
        labels = labels.to(device)

        output = model(images.float())

        train_loss = criterion(output, labels)

        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(validate_loader):
            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            predictions_validate.append(output)
            labels_validate.append(labels)

        for i, (images, labels) in enumerate(test_loader):
            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images.float())
            test_loss = criterion(output, labels)

            predictions_test.append(output)
            labels_test.append(labels)

    train_loss_epoch = running_train_loss / len(combined_loader)
    validate_balanced_accuracy, _ = balanced_accuracy(labels_validate, predictions_validate)
    validate_overall_accuracy = overall_accuracy(labels_validate, predictions_validate)
    test_balanced_accuracy, _ = balanced_accuracy(labels_test, predictions_test)
    test_overall_accuracy = overall_accuracy(labels_test, predictions_test)

    stats_epoch = {
        'epoch': f'{epoch + 1}',
        'train loss': f'{train_loss_epoch:.3f}',
        'validate balanced accuracy': f'{validate_balanced_accuracy * 100:.2f}%',
        'validate overall accuracy': f'{validate_overall_accuracy * 100:.2f}%',
        'test balanced accuracy': f'{test_balanced_accuracy * 100:.2f}%',
        'test overall accuracy': f'{test_overall_accuracy * 100:.2f}%'
    }

    stats.append(stats_epoch)

    fieldnames = ['epoch', 'train loss', 'validate balanced accuracy', 'validate overall accuracy',
                  'test balanced accuracy', 'test overall accuracy']

    with open('stats_interpolation_label_as_file.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    if validate_balanced_accuracy > highest_evaluation_accuracy:
        highest_evaluation_accuracy = validate_balanced_accuracy
        torch.save(model.state_dict(), 'checkpoints_interpolation_label_as_file/checkpoint highest accuracy.pt')

    elif (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoints_interpolation_label_as_file/checkpoint epoch {}.pt'.format(epoch + 1))
