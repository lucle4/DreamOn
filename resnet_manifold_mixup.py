import os
import csv
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('benign', 'malignant', 'normal')

n_epochs = 100
batch_size = 20
lr = 0.001
alpha = 0.8

img_size = 256
n_classes = len(classes)

directory = os.getcwd()
img_dir_original = os.path.join(directory, 'BUSI/split/train')
label_dir_original = os.path.join(directory, 'BUSI/split/labels_train.csv')

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


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def register_forward_hook(layer):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    handle = layer.register_forward_hook(hook)

    return activations, handle


def continue_forward_pass(model, layer_name, x):
    if layer_name == 'layer1':
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
    elif layer_name == 'layer2':
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
    elif layer_name == 'layer3':
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
    elif layer_name == 'layer4':
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
    elif layer_name == 'fc':
        pass

    return x


transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])


train_dataset = CustomDataset(img_dir_original, label_dir_original, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = CustomDataset(img_dir_validate, label_dir_validate, transform=transform)
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

test_dataset = CustomDataset(img_dir_test, label_dir_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

print(model)

if torch.cuda.is_available():
    model.cuda()

eligible_layers = {'layer1': model.layer1,
                   'layer2': model.layer2,
                   'layer3': model.layer3,
                   'layer4': model.layer4,
                   'fc': model.fc}

hook_outputs = {name: register_forward_hook(layer) for name, layer in eligible_layers.items()}

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

    # no mixup
    for i, (images, labels) in enumerate(train_loader):
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

    # with mixup
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        for name, (activations, _) in hook_outputs.items():
            del activations[:]

        images = images.to(device)
        labels = labels.to(device)

        output = model(images.float())

        selected_layer_name = np.random.choice(list(eligible_layers.keys()))

        activations, _ = hook_outputs[selected_layer_name]
        activations = activations[0]

        mixed_activations, labels_a, labels_b, lam = mixup_data(activations, labels, alpha)
        mixed_output = continue_forward_pass(model, selected_layer_name, mixed_activations)

        loss_a = criterion(mixed_output, labels_a)
        loss_b = criterion(mixed_output, labels_b)
        train_loss = lam * loss_a + (1 - lam) * loss_b

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

    train_loss_epoch = running_train_loss / (len(train_loader) * 2)
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

    with open('stats_manifold_mixup.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for parameter in stats:
            writer.writerow(parameter)

    if validate_balanced_accuracy > highest_evaluation_accuracy:
        highest_evaluation_accuracy = validate_balanced_accuracy
        torch.save(model.state_dict(), 'checkpoints_manifold_mixup/checkpoint highest accuracy.pt')

    elif (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoints_manifold_mixup/checkpoint epoch {}.pt'.format(epoch + 1))
