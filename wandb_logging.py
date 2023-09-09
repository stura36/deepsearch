from kaggle_secrets import UserSecretsClient
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_logger_hparams(logger, model, datamodule):

    logger.experiment.config["batch_size"] = datamodule.batch_size
    logger.experiment.config["learning_rate"] = model.lr
    logger.experiment.config["train size"] = datamodule.get_train_size()
    logger.experiment.config["validation size"] = datamodule.get_val_size()
    logger.experiment.config["test size"] = datamodule.get_test_size()
    logger.experiment.config["random_caption"] = datamodule.random_caption

    return


def setup_wandb(model, data_module):
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("wandb_api_key")

    wandb.login(key=wandb_api_key)
    wandb_logger = WandbLogger(project="diplomski")

    setup_logger_hparams(wandb_logger, model, data_module)

    checkpoint_callback = ModelCheckpoint(monitor="r1txt_val", mode="max")
    return wandb_logger,checkpoint_callback



