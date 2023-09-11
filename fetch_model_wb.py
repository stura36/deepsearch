import wandb
import os

def fetch_model_from_wandb(target_dir = "artifacts"):
    api = wandb.Api()
    model_art = api.artifact("stura36/diplomski/model-nn1dpgyb:v0", type = "model")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    model_art.download(target_dir)
    return

if __name__ == '__main__':
    fetch_model_from_wandb()
