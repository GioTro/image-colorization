from utils import Utils
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from model import ColorizationModel
from torch.nn import BatchNorm2d
from torch import device, cuda, profiler

DEVICE : device

def pre_trained(model: ColorizationModel) -> ColorizationModel:
    """
    Since the architecture is the same as https://github.com/richzhang/colorization
    We can use their pre-trained model.
    """
    model.load_state_dict(
        model_zoo.load_url(
            "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
            map_location="cpu",
            check_hash=True,
        )
    )
    return model


def get_trainer(trainer_param: dict, callbacks: dict) -> Trainer:
    if len(callbacks) != 0:
        clb = [c(**callbacks[c]) for c in callbacks.keys()]
        trainer_param["callbacks"] = clb

    trainer = Trainer(**trainer_param)
    return trainer


def get_model(dic: dict, pretrained=False) -> ColorizationModel:
    assert len(dic) == 1
    model_class = list(dic.keys())[0]
    model = model_class(**dic[model_class])
    if pretrained:
        model = pre_trained(model)
    return model


def get_param() -> dict:
    # fmt: off
    param = {
        'trainer' : {
            'max_epochs' : 100,
            'log_every_n_steps' : 20,
            'limit_train_batches' : 1.0,
            'limit_val_batches' : 1.0,
            'gpus' : 1,
            'check_val_every_n_epoch' : 1,
        },
        'callbacks' : {
            EarlyStopping : {
                'monitor' : 'val_loss_epoch',
                'min_delta' : 0.0,
                'check_finite' : True,
                'patience' : 7,
                'verbose' : True,
                'check_on_train_epoch_end' : True,
                'mode' : 'min',
            },
            pl.callbacks.LearningRateMonitor : {
                'logging_interval' : 'epoch',
            },
            ModelCheckpoint : {
                'save_last' : True,
                'verbose' : True,
                'every_n_val_epochs' : 1,
            }

        },
        'model' : {
            ColorizationModel : {
                'num_workers' : 6,
                'norm_layer' : BatchNorm2d,
                'batch_size' : 64,
                'T_max': 1e5, 
                'eta_min': 1e-7,
                'optimizer_param' : {
                    'Adam': {
                        'lr': 3e-4, 
                        'betas': (.90, .99),
                        'weight_decay': 1e-3,
                    },
                }
            },
        },
        'pretrained' : False,
        'cpu' : False,
    }
    # fmt: on
    return param


if __name__ == "__main__":
    param = get_param()

    if param['cpu']:
        param['trainer']['gpus'] = 0


    model = get_model(param["model"], pretrained=param['pretrained'])
    trainer = get_trainer(param["trainer"], param["callbacks"])

    trainer.fit(model)
