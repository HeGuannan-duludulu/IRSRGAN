import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.many_blocks = [DenseResidualBlock(in_channels) for _ in range(3)]
        self.rrdb = nn.Sequential(*self.many_blocks)

        self.conv = nn.Conv2d(256, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        final_result = self.rrdb(x) * self.residual_beta
        temp = x
        for step in self.many_blocks:
            temp = step(temp) * self.residual_beta
            final_result = torch.cat((final_result, temp), 1)
        final_result = self.conv(final_result)

        aaa = final_result + x
        # aaa = self.rrdb(x) * self.residual_beta + x
        return aaa


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()
        self.initial = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels),
            UpsampleBlock(num_channels),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)




class IDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(IDiscriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F.l1_loss(sr_feature, gt_feature)

        return loss


def content_loss(feature_model_extractor_node,
                 feature_model_normalize_mean,
                 feature_model_normalize_std) -> ContentLoss:
    content_loss = ContentLoss(feature_model_extractor_node,
                               feature_model_normalize_mean,
                               feature_model_normalize_std)

    return content_loss


def test():
    gen = Generator()
    low_res = 32
    x = torch.randn((5, 3, low_res, low_res))
    print(x.shape)
    gen_out = gen(x)
    i_disc = IDiscriminator()
    disc_out = i_disc(gen_out)
    print(disc_out)


if __name__ == "__main__":
    test()
