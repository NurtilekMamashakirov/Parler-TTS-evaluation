import io
import random

import pandas as pd
import requests
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, length, random_seed=None):
        super(AudioDataset, self).__init__()
        self.X = pd.DataFrame(columns=['transcription', 'transcription_normalised', 'text_description'])
        self.y = []
        if random_seed is not None:
            random.seed(random_seed)
        offset = random.randint(0, 20000)
        rows = requests.get(
            url=f"https://datasets-server.huggingface.co/rows?dataset=reach-vb%2Fjenny_tts_dataset"
                f"&config=default&split=train&offset={offset}&length={length}"
        ).json()['rows']
        rows_with_description = requests.get(
            url=f"https://datasets-server.huggingface.co/rows?dataset=ylacombe%2Fjenny-tts-10k-tagged&config=default"
                f"&split=train&offset={offset}&length={length}"
        ).json()['rows']
        for i in range(length):
            audio_response = requests.get(url=rows[i]['row']['audio'][0]['src'])
            audio_data = sf.read(io.BytesIO(audio_response.content))[0]
            self.y.append(audio_data)
            transcription = rows[i]['row']['transcription']
            transcription_normalized = rows[i]['row']['transcription_normalised']
            text_description = rows_with_description[i]['row']['text_description']
            self.X[i] = [transcription, transcription_normalized, text_description]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        row = self.X[index]
        dict_row = {"transcription": row[0], "transcription_normalised": row[1], "text_description": row[2]}
        return dict_row, self.y[index]


def generate_prompts_with_gigachat(prompts_quantity):
    pass


def generate_describes_with_gigachat(describes_quantity):
    pass
