import os
import subprocess
from loguru import logger
from pydantic import BaseModel, Field

jobs = {}
processes = {}

class TrainingParameters(BaseModel):
    pretrained_model: str = Field(..., description="Имя или путь к предобученной модели (например, stabilityai/stable-diffusion-xl-base-1.0)")
    pretrained_vae: str = Field(..., description="Имя или путь к VAE модели (например, madebyollin/sdxl-vae-fp16-fix)")
    dataset_name: str = Field(..., description="Имя датасета (например, Sandrro/genbuilder_data)")
    resolution: int = Field(default=512, description="Разрешение изображений")
    center_crop: bool = Field(default=True, description="Флаг центрированного кропа")
    random_flip: bool = Field(default=True, description="Флаг случайного разворота изображений")
    proportion_empty_prompts: float = Field(default=0.2, description="Доля пустых промптов")
    train_batch_size: int = Field(default=1, description="Размер батча для обучения")
    gradient_accumulation_steps: int = Field(default=4, description="Количество шагов аккумуляции градиентов")
    gradient_checkpointing: bool = Field(default=True, description="Использование градиент чекпоинтинга")
    max_train_steps: int = Field(default=10000, description="Максимальное число шагов обучения")
    use_8bit_adam: bool = Field(default=True, description="Использование 8-bit Adam оптимизатора")
    learning_rate: float = Field(default=1e-06, description="Начальная скорость обучения")
    lr_scheduler: str = Field(default="constant", description="Тип scheduler'а для скорости обучения")
    lr_warmup_steps: int = Field(default=0, description="Количество шагов для разогрева (warmup)")
    mixed_precision: str = Field(default="fp16", description="Режим смешанной точности")
    report_to: str = Field(default="wandb", description="Система для отчётности (например, wandb)")
    validation_prompt: str = Field(
        default="overhead vector_map, residential, School, Polyclinic, Park, density_2, 1.0 km²",
        description="Промпт для валидации модели"
    )
    validation_epochs: int = Field(default=5, description="Частота валидации (через N эпох)")
    checkpointing_steps: int = Field(default=5000, description="Шаги для сохранения чекпоинтов")
    output_dir: str = Field(default="genbuilder", description="Директория для сохранения результатов")
    push_to_hub: bool = Field(default=True, description="Флаг для публикации модели в Hub")

def run_training(job_id: str, params: TrainingParameters):
    env = os.environ.copy()
    env["MODEL_NAME"] = params.pretrained_model
    env["VAE_NAME"] = params.pretrained_vae
    env["DATASET_NAME"] = params.dataset_name

    command = [
        "accelerate", "launch", "app/logic/train_text_to_image_sdxl.py",
        "--pretrained_model_name_or_path", params.pretrained_model,
        "--pretrained_vae_model_name_or_path", params.pretrained_vae,
        "--dataset_name", params.dataset_name,
        "--resolution", str(params.resolution)
    ]
    if params.center_crop:
        command.append("--center_crop")
    if params.random_flip:
        command.append("--random_flip")
    command.extend([
        "--proportion_empty_prompts", str(params.proportion_empty_prompts),
        "--train_batch_size", str(params.train_batch_size),
        "--gradient_accumulation_steps", str(params.gradient_accumulation_steps)
    ])
    if params.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    command.extend([
        "--max_train_steps", str(params.max_train_steps),
        "--use_8bit_adam" if params.use_8bit_adam else "",
        "--learning_rate", str(params.learning_rate),
        "--lr_scheduler", params.lr_scheduler,
        "--lr_warmup_steps", str(params.lr_warmup_steps),
        "--mixed_precision", params.mixed_precision,
        "--report_to", params.report_to,
        "--validation_prompt", params.validation_prompt,
        "--validation_epochs", str(params.validation_epochs),
        "--checkpointing_steps", str(params.checkpointing_steps),
        "--output_dir", params.output_dir
    ])
    if params.push_to_hub:
        command.append("--push_to_hub")
    
    command = [arg for arg in command if arg != ""]

    try:
        jobs[job_id] = "Running"
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   env=env,
                                   text=True,
                                   bufsize=1)
        processes[job_id] = process

        for line in process.stdout:
            logger.info(line.rstrip())

        return_code = process.wait()
        if return_code != 0:
            jobs[job_id] = f"Failed: Return code {return_code}"
        else:
            jobs[job_id] = "Finished"
    except subprocess.CalledProcessError as e:
        jobs[job_id] = f"Failed: {str(e)}"
    except Exception as e:
        jobs[job_id] = f"Failed: {str(e)}"
    finally:
        processes.pop(job_id, None)