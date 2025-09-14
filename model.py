import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ImageCompressionModel(pl.LightningModule):
    def __init__(self, latent_dim=256, learning_rate=1e-3):
        super(ImageCompressionModel, self).__init__()
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self.activation_func = nn.LeakyReLU()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        e1 = self.activation_func(self.encoder[0](x))
        e2 = self.activation_func(self.encoder[1](e1))
        e3 = self.activation_func(self.encoder[2](e2))

        d1 = self.activation_func(self.decoder[0](e3) + e2)

        d2 = self.activation_func(self.decoder[1](d1) + e1)  # skip from middle encoder
        d3 = self.decoder[2](d2) + x  # skip from first encoder

        x_recon = torch.sigmoid(d3)
        return x_recon

    def training_step(self, batch, batch_idx):
        x, target = batch

        x_recon = self.forward(x)
        mse_loss = F.mse_loss(x_recon, target)

        loss = mse_loss

        self.log("train_loss", loss)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
                "name": "learning_rate",
            },
        }


def main() -> None:
    import torchvision.transforms as transforms
    from torchsummary import summary

    model = ImageCompressionModel()

    def image_to_tensor(image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image).unsqueeze(0)

    image_tensor = image_to_tensor("image.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    summary(model, image_tensor.shape[1:])

    with torch.no_grad():
        reconstructed = model(image_tensor)

    print("Reconstructed shape:", reconstructed.shape)


if __name__ == "__main__":
    main()
