import wandb
from src.train import train
from src.sweep_config import sweep_config

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="inat-hyper-sweep")
    wandb.agent(sweep_id, function=train, count=5)
