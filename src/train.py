from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from numpy.core.fromnumeric import argmax
import torch.utils.model_zoo as model_zoo
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping
from model import ColorizationModel
import parser
from torch.nn import BatchNorm2d

def pre_trained(model : ColorizationModel) -> ColorizationModel:
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

def get_trainer(trainer_param : dict, callbacks : dict) -> Trainer:
    if len(callbacks) != 0:
        clb = [c(**callbacks[c]) for c in callbacks.keys()]
        trainer_param['callbacks'] = clb
    
    trainer = Trainer(**trainer_param)
    return trainer

def get_model(dic : dict, pretrained = False) -> ColorizationModel:
    assert len(dic) == 1
    model_class = list(dic.keys())[0]
    model = model_class(**dic[model_class])
    if pretrained:
        model = pre_trained(model)
    return model

def get_default() -> dict:
    # fmt: off
    default_param = {
        'trainer' : {
            'max_epochs' : 100,
            'gpus' : 1,
            'log_every_n_steps' : 20,
            'limit_train_batches' : 1.0,
            'limit_val_batches' : 1.0,
            'check_val_every_n_epoch' : 1,
            'callbacks' : [],
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
        }
    }
    # fmt: on
    return default_param

if __name__ == "__main__":
    bypass_parser = False
    if bypass_parser:
        param = get_default()
        param['model'] = get_model(param['model'])
    else:
        param = parser.process()

    trainer = get_trainer(param['trainer'], param['callbacks'])
    trainer.fit(param['model'])