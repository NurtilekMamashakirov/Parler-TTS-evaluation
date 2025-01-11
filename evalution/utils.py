import json
import os
import random
import uuid

import librosa
import requests
import soundfile as sf
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def prepare_dataset(tests_quantity: int, random_seed=None):
    dataset = load_dataset("reach-vb/jenny_tts_dataset", split='train', )
    dataset_with_description = load_dataset("ylacombe/jenny-tts-10k-tagged", split='train')
    if random_seed:
        random.seed(random_seed)
    offset = random.randint(0, len(dataset) - tests_quantity)
    return dict(dataset[offset:offset + tests_quantity], **dataset_with_description[offset:offset + tests_quantity])


def receive_audios(pred_audio: torch.Tensor, target_audio: torch.Tensor, cut_audio: bool = True):
    target_audio = torch.from_numpy(librosa.resample(target_audio.cpu().numpy(), orig_sr=48000, target_sr=16000)).to(
        device)
    pred_audio = torch.from_numpy(librosa.resample(pred_audio.cpu().numpy(), orig_sr=44100, target_sr=16000)).to(
        device)
    if cut_audio:
        min_len = min(len(pred_audio), len(target_audio))
        pred_audio = pred_audio[:min_len]
        target_audio = target_audio[:min_len]
    return pred_audio, target_audio


def generate_prompts_with_gigachat(prompts_quantity):
    access_token = get_access_token()
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    prompt = (f"Generate {prompts_quantity} sentences with 10-15 words, "
              f"each sentence must start on a new line. Don't use any numbers in the answer!")
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {"role": "user",
             "content": prompt}
        ],
        "temperature": 0.5,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    sentences = \
        requests.request("POST", url, headers=headers, data=payload, verify=False).json()['choices'][0][
            'message']['content'].split('\n')
    return sentences


def generate_describes_with_gigachat(describes_quantity):
    access_token = get_access_token()
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    prompt = (f"Generate {describes_quantity} sentences. Each sentence must describe speech's quality by 3 parameters:"
              f"noise (quite clear, very clear, slightly clear, moderate ambient sound, slightly noisy, quite noisy), "
              f"reverberation (very confined sound, quite confined sounding, slightly confined sounding), "
              f"speech_monotony (quite monotone, slightly monotone, very monotone). Each sentence must be on a new line."
              f"Don't write anything else except of that sentences!")
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [
            {"role": "user",
             "content": prompt}
        ],
        "temperature": 0.5,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    sentences = \
        requests.request("POST", url, headers=headers, data=payload, verify=False).json()['choices'][0][
            'message']['content'].split('\n')
    return sentences


def get_access_token():
    load_dotenv()
    authorization_key = os.environ.get("GIGACHAT_KEY")
    auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    rq_uid = str(uuid.uuid4())
    payload = {
        'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': rq_uid,
        'Authorization': f'Basic {authorization_key}'
    }
    access_token = requests.request("POST", auth_url, headers=headers, data=payload, verify=False).json()[
        'access_token']
    return access_token


def save_audio(audio: np.ndarray, sampling_rate: int):
    directory_name = "audios"
    if not os.path.exists(directory_name) or not os.path.isdir(directory_name):
        os.mkdir(directory_name)
    audios_in_directory = os.listdir(directory_name)
    sf.write(f"{directory_name}/audio{len(audios_in_directory)}.wav", audio, sampling_rate)
