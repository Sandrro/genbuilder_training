from fastapi import FastAPI
import wandb
from huggingface_hub import login
from app.dependencies import config, setup_logger
from app.routers.logs_routers import logs_router
from app.routers.training_routers import training_router

hf_token = config.get("HUGGINGFACE_HUB_TOKEN")
wandb_key = config.get("WANDB_API_KEY")

login(token=hf_token)
wandb.login(key=wandb_key)

setup_logger(config)

app = FastAPI(title="GenBuilder Training")

app.include_router(logs_router)

app.include_router(training_router, prefix="/training")