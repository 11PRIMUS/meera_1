import os
from openai import OpenAI
from pathlib import Path
from typing import Optional
import time

NEBUIS_BASE_URL=""
TRAINING_FILE=Path("tune/meera(10004),jsonl")

#no validation
VALIDATION_FILE:Optional[Path] = None

MODEL_NAME=""
SUFFIX=""
POLL_INTERVAL_SECONDS=15
CHECKPOINT_DIR = Path("checkpoints")

def get_client()->OpenAI:
    api_key=os.environ.get("NEBUIS_API_KEY")
    if not api_key:
        raise RuntimeError("check NEBUIS api key")
    return OpenAI(base_url=NEBUIS_BASE_URL, api_key=api_key)

def upload_dataset(client:OpenAI, path :Path)->str:
    if not path or not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with path.open("rb") as handle:
        dataset = client.files.create(file=handle, purpose="fine-tune")
    print(f"uploaded {path} → {dataset.id}")
    return dataset.id

def build_job_request(training_file_id: str, validation_file_id: str | None) -> dict:
    job_request: dict[str, object] = {
        "model": MODEL_NAME,
        "training_file": training_file_id,
        "suffix": SUFFIX,
        "hyperparameters": {
            "batch_size": 8,
            "learning_rate": 1e-5,
            "n_epochs": 3,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "lora": True,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "packing": True,
            "max_grad_norm": 1.0,
            "context_length": 8192,
        },
        "seed": 42,
    }
    if validation_file_id:
        job_request["validation_file"] = validation_file_id
    return job_request

def poll_job(client:OpenAI, job_id:str)->object:
    terminal_statuses = {"succeeded", "failed", "cancelled"}
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"[status] {job.status}")
    while job.status not in terminal_statuses:
        time.sleep(POLL_INTERVAL_SECONDS)
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"[status] {job.status}")
    return job

def print_events(client: OpenAI, job_id: str) -> None:
    events = client.fine_tuning.jobs.list_events(job_id)
    print("\n job events")
    for event in events.data:
        print(f"{event.created_at} {event.level} - {event.message}")

def download_checkpoints(client: OpenAI, job_id: str) -> None:
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id).data
    if not checkpoints:
        print("no checkpoints returned")
        return

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for checkpoint in checkpoints:
        checkpoint_dir = CHECKPOINT_DIR / checkpoint.id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading checkpoint {checkpoint.id} (step {checkpoint.step_number})")
        for file_id in checkpoint.result_files:
            file_meta = client.files.retrieve(file_id)
            filename = Path(file_meta.filename or file_id).name
            destination = checkpoint_dir / filename
            content = client.files.content(file_id)
            content.write_to_file(str(destination))
            print(f"  saved {destination}")


def main()->None:
    client=get_client()
    training_file_id = upload_dataset(client, TRAINING_FILE)
    validation_file_id = upload_dataset(client, VALIDATION_FILE) if VALIDATION_FILE else None

    job_request = build_job_request(training_file_id, validation_file_id)
    print("fine-tuning job with payload:")
    print(job_request)

    job =client.fine_tuning.jobs.create(**job_request)
    print(f"Created job {job.id} (status: {job.status})")

    job = poll_job(client, job.id)
    print(f"final status: {job.status}")
    if job.status == "failed":
        print(f"job failed with error: {job.error}")
        return

    print_events(client, job.id)
    download_checkpoints(client, job.id)


if __name__ =="__main__":
    main()
