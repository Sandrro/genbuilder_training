from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.logic.training_logic import TrainingParameters, run_training, jobs, processes
import uuid

training_router = APIRouter()

@training_router.post("/train", summary="Запустить обучение модели")
async def train_model(params: TrainingParameters, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_training, job_id, params)
    return {"job_id": job_id, "status": "Started"}

@training_router.get("/job/{job_id}", summary="Статус задачи обучения")
async def get_job_status(job_id: str):
    status = jobs.get(job_id, "Job not found")
    return {"job_id": job_id, "status": status}

@training_router.get("/jobs", summary="Список всех задач обучения")
async def get_all_jobs():
    return jobs

@training_router.delete("/cancel/{job_id}", summary="Остановить выполнение задачи обучения")
async def cancel_job(job_id: str):
    process = processes.get(job_id)
    if process is None:
        raise HTTPException(status_code=404, detail=f"Job not found or already finished: {job_id}")

    try:
        process.kill()
        jobs[job_id] = "Cancelled"
        return {"job_id": job_id, "status": "Cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")
