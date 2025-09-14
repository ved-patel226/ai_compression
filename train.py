from model import ImageCompressionModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

pl.seed_everything(42)

import torch
import os
from pytorch_msssim import ssim


def image_to_tensor(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)


image_path = "image.png"
input_tensor = image_to_tensor(image_path)
target_tensor = input_tensor.clone()


batch_size = 1
num_epochs = 1000
learning_rate = 1e-3


model = ImageCompressionModel(latent_dim=512, learning_rate=learning_rate)


class ProgressCheckCallback(pl.Callback):
    def __init__(self):
        super().__init__()

        self.save_dir = "./imgs/"

        i = 0
        while os.path.exists(f"{self.save_dir}/{i}"):
            i += 1
        self.save_dir = f"{self.save_dir}/{i}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.i = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.i += 1

        if self.i % 10 != 0:
            return

        with torch.no_grad():
            device_input = input_tensor.to(pl_module.device)
            sample_recon = pl_module(device_input)

            score = (
                ssim(sample_recon.unsqueeze(0), device_input.unsqueeze(0)).item() * 100
            )

            recon_image = transforms.ToPILImage()(sample_recon.cpu().squeeze(0))
            recon_image.save(f"{self.save_dir}/{trainer.current_epoch}_{score:.2f}.png")


trainer = pl.Trainer(
    max_epochs=num_epochs,
    logger=TensorBoardLogger("logs/"),
    log_every_n_steps=10,
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="image_compression-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
        ),
        ProgressCheckCallback(),
    ],
)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(input_tensor, target_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size)

trainer.fit(model, train_loader)
