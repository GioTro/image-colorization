import torch
from torch.functional import F
import pytorch_lightning as pl
import dataloader as dl
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BatchNorm2d


class ColorizationModel(pl.LightningModule):
    def __init__(
        self,
        norm_layer=BatchNorm2d,
        num_workers=6,
        batch_size=128,
        T_max=39000,
        eta_min=1e-7,
        optimizer_param={"Adam": {"lr": 3e-4, "betas": 0.95, "weight_decay": 1e-3}},
    ):
        super(ColorizationModel, self).__init__()
        self.T_max = T_max
        self.batch_size = batch_size
        self.eta_min = eta_min
        self.optimizer_param = optimizer_param

        # fmt: off
        self.data_loaders = dl.return_loaders(
            batch_size=batch_size, num_workers=num_workers, shuffle=True
        )

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), 
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128)
        )
        
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )        

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )

        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.model_out = nn.Conv2d(
            313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False
        )
        # fmt:on

        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.normalize_l = lambda x: (x - 50) / 100
        self.unnormalize_l = lambda x: (x + 50) * 100

        self.normalize_ab = lambda x: x / 110
        self.normalize_ab = lambda x, y: (x/110, y/110)
        self.unnormalize_ab = lambda x: x * 110

    def forward(self, X):
        conv1 = self.model1(self.normalize_l(X))
        conv2 = self.model2(conv1)
        conv3 = self.model3(conv2)
        conv4 = self.model4(conv3)
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)
        logit = self.model8(conv7)
        out = self.model_out(self.softmax(logit))
        return self.upsample(self.unnormalize_ab(out))

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = F.mse_loss(*self.normalize_ab(y_pred, y))
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = F.mse_loss(*self.normalize_ab(y_pred, y))
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = F.mse_loss(*self.normalize_ab(y_pred, y))
        self.log(
            "test_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), **self.optimizer_param["Adam"])
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, eta_min=self.eta_min, T_max=self.T_max
        )
        return [optimizer], [scheduler]

    def predict_step(self, batch: int, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def on_epoch_end(self):
        global_step = self.global_step
        # self.print_photos(self.current_epoch)
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, global_step)

    def train_dataloader(self):
        return self.data_loaders["train"]

    def test_dataloader(self):
        return self.data_loaders["test"]

    def val_dataloader(self):
        return self.data_loaders["validation"]

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)