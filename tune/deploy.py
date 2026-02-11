import os
import sys
from typing import List
import requests
import time
from openai import OpenAI
from dotenv import load_dotenv


#out of nebuis credit, earlier custom model are deleted, when nebuis shifted to nebuistokenfactory
#will be using meta-llama/Llama-3.3-70B-Instruct having context length 8,192, for chat, will switch to original base meera when i have credits
api_key=os.environ.get('NEBUIS_API_KEY')
api_url="https://api.tokenfactory.nebius.com/"
client=OpenAI(
    base_url=api_url+"/v1",
    api_key=api_key,)
BASE_MODEL="Qwen/Qwen3-32B"
ADAPTER_NAME="meeradapter"
FINE_TUNE_JOB_ID="ftjob-3edb9960a2194fa7908e536bf8611111"
FINE_TUNE_CHECKPOINT_ID="ftckpt_40107b28-f083-46d2-8aee-b838f03cfefb"
POLL_INTERVAL_SECONDS=10

SYSTEM_PROMPT = (
    "You are Meera, an empathetic journaling companion. "
    "Respond with warmth, validation, and gentle prompts that help users reflect."
)
SAMPLE_HISTORY:List[dict[str, str]]=[
    {"role": "user", "content": "Hi Meera, it was a heavy day but I made it through."},
    {"role": "assistant", "content": "I'm here with you. What part felt the heaviest?"},
]
SAMPLE_MESSAGE = "Thank you for checking in. I just want to feel grounded again."

def create_lora_from_job(api_key: str) -> str:
    payload = {
        "source": f"{FINE_TUNE_JOB_ID}:{FINE_TUNE_CHECKPOINT_ID}",
        "base_model": BASE_MODEL,
        "name": ADAPTER_NAME,
        "description": "meera qwen deployment",
    }
    response = requests.post(
        f"{api_url}/v0/models",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    model_name = response.json().get("name")
    if not model_name:
        raise RuntimeError(f"Deployment response missing model name: {response.text}")
    print(f"Started deployment for adapter '{ADAPTER_NAME}'. Model name: {model_name}")
    return model_name


def wait_for_validation(api_key: str, model_name: str) -> dict:
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        response = requests.get(
            f"{api_url}/v0/models/{model_name}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        info = response.json()
        status = info.get("status")
        print(f"[deploy] {model_name} status: {status}")
        if status in {"active", "error"}:
            return info


def send_sample_completion(api_key: str, model_name: str) -> str:
    client = OpenAI(base_url=api_url, api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *SAMPLE_HISTORY]
    messages.append({"role": "user", "content": SAMPLE_MESSAGE})

    completion = client.chat.completions.create(model=model_name, messages=messages)
    choice = completion.choices[0]
    if not choice.message or not choice.message.content:
        raise RuntimeError("Completion response did not include message content.")
    return choice.message.content.strip()


def main() -> None:
    if "<ftjob-" in FINE_TUNE_JOB_ID or "<ftckpt-" in FINE_TUNE_CHECKPOINT_ID:
        print("set fine_tune id and fine tune checkpoint before running.")
        sys.exit(1)

    api_key =api_key
    model_name = create_lora_from_job(api_key)
    status_info = wait_for_validation(api_key, model_name)

    if status_info.get("status") == "error":
        reason = status_info.get("status_reason", "unknown error")
        print(f"Deployment failed: {reason}")
        sys.exit(1)

    reply = send_sample_completion(api_key, model_name)
    print("\n=== Meera reply ===")
    print(reply)
    print(f"\nSet NEBIUS_MODEL={model_name} so the backend uses this fine-tuned model.")


if __name__ == "__main__":
    main()
