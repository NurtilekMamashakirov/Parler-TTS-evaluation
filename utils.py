import random

import librosa
import torch
from datasets import Audio
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def prepare_dataset(tests_quantity: int, random_seed=None):
    dataset = load_dataset("reach-vb/jenny_tts_dataset", split='train', )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset_with_description = load_dataset("ylacombe/jenny-tts-10k-tagged", split='train')
    if random_seed:
        random.seed(random_seed)
    offset = random.randint(0, len(dataset) - tests_quantity)
    return dict(dataset[offset:offset + tests_quantity], **dataset_with_description[offset:offset + tests_quantity])


def receive_audios(pred_audio: torch.Tensor, target_audio: torch.Tensor, cut_audio: bool = True):
    target_audio = target_audio.to(device)
    pred_audio = torch.from_numpy(librosa.resample(pred_audio.cpu().numpy(), orig_sr=44100, target_sr=16000)).to(
        device)
    if cut_audio:
        min_len = min(len(pred_audio), len(target_audio))
        pred_audio = pred_audio[:min_len]
        target_audio = target_audio[:min_len]
    return pred_audio, target_audio


def generate_prompts_with_gigachat(prompts_quantity):
    pass


def generate_describes_with_gigachat(describes_quantity):
    pass
