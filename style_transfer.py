import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19(pretrained=True).features


def transform(images):
    images = images.repeat(1, 3, 1, 1)
    images = images.squeeze(1)

    inv_normalization = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                 std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                            transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                 std=[1., 1., 1.]), ])

    normalization = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    images = inv_normalization(images)
    images = normalization(images)

    return images


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


model = VGG().to(device).eval()


alpha = 1
beta = 0.01


def run_style_transfer(style_img, content_img):
    content_img = transform(content_img)
    style_img = transform(style_img)

    generated_content = style_img.clone().requires_grad_(True)
    generated_style = content_img.clone().requires_grad_(True)

    generated_features_content = model(generated_content)
    generated_features_style = model(generated_style)

    original_img_features = model(content_img)
    style_features = model(style_img)
    style_loss = content_loss = 0

    for gen_feature_content, gen_feature_style, orig_feature, style_features in zip(
            generated_features_content, generated_features_style, original_img_features, style_features):
        batch_size, channel, height, width = gen_feature_content.shape

        content_loss += torch.mean((gen_feature_content - orig_feature) ** 2)

        G = gen_feature_style.view(batch_size * channel, height * width).mm(
            gen_feature_style.view(batch_size * channel, height * width).t())

        A = style_features.view(batch_size * channel, height * width).mm(
            style_features.view(batch_size * channel, height * width).t())

        style_loss += torch.mean((G - A) ** 2)

    style_loss = beta * style_loss
    content_loss = alpha * content_loss

    return style_loss.item(), content_loss.item()
