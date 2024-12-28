import numpy as np
import torch
from jiwer import wer
from matplotlib import pyplot as plt
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility, \
    ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

from utils import receive_audios

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DNSMOSEvaluator:
    def __init__(self, model_tts, tokenizer):
        super().__init__()
        self.model_tts = model_tts.to(
            device)
        self.tokenizer = tokenizer
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=self.model_tts.config.sampling_rate, personalized=False)
        self.dnsmos_results = []

    def evaluate(self, prompt: str, description: str, save_audio: bool = False):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(device)
        metric = self.dnsmos(audio)
        self.dnsmos_results.append(metric)
        return metric

    def visualize(self):
        self.dnsmos.plot(self.dnsmos_results)
        plt.show()

    def remove_history(self):
        self.dnsmos_results = []


class WEREvaluator:
    def __init__(self, model_tts, tokenizer, pipeline_stt):
        self.model_tts = model_tts.to(device)
        self.tokenizer = tokenizer
        self.pipeline_stt = pipeline_stt
        self.word_error_rates = []

    def evaluate(self, prompt: str, description: str, save_audios: bool = False):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).cpu().numpy().squeeze()
        hypothesis_prompt = self.pipeline_stt(audio)["text"]
        word_error_rate = wer(reference=prompt, hypothesis=hypothesis_prompt)
        self.word_error_rates.append(word_error_rate)
        return word_error_rate

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.word_error_rates)), self.word_error_rates)
        plt.title('Word Error Rate Evaluation')
        plt.xlabel('tests')
        plt.ylabel('scores')
        plt.xticks(np.arange(len(self.word_error_rates)))
        plt.grid()
        plt.show()

    def remove_history(self):
        self.word_error_rates = []


class ObjectiveMetricsEvaluator:
    def __init__(self, model_tts, tokenizer):
        self.model_tts = model_tts
        self.tokenizer = tokenizer
        self.perceptual_evaluation_of_speech_qualities = []
        self.scale_invariant_signal_to_distortion_ratios = []
        self.short_time_objective_intelligibility = []
        self.pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb").to(device)
        self.si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
        self.stoi_metric = ShortTimeObjectiveIntelligibility(fs=16000, extended=True).to(device)

    def evaluate(self, prompt: str, description: str, target_audio: torch.Tensor, save_audio: bool = False):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        pred_audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(
            device).squeeze(0)
        target_audio = target_audio.to(device)
        pred_audio, target_audio = receive_audios(pred_audio, target_audio)

        pesq = self.pesq_metric(pred_audio, target_audio)
        self.perceptual_evaluation_of_speech_qualities.append(pesq)

        si_sdr = self.si_sdr_metric(pred_audio, target_audio)
        self.scale_invariant_signal_to_distortion_ratios.append(si_sdr)

        stoi = self.stoi_metric(pred_audio, target_audio)
        self.short_time_objective_intelligibility.append(stoi)
        return {'pesq': pesq, 'si_sdr': si_sdr, 'stoi': stoi}

    def visualize(self):
        self.pesq_metric.plot(self.perceptual_evaluation_of_speech_qualities)
        plt.show()
        self.si_sdr_metric.plot(self.scale_invariant_signal_to_distortion_ratios)
        plt.show()
        self.stoi_metric.plot(self.short_time_objective_intelligibility)
        plt.show()


class SimilarityEvaluator:
    def __init__(self, model_tts, tokenizer, model_audio2vec, feature_extractor):
        self.model_tts = model_tts
        self.tokenizer = tokenizer
        self.model_audio2vec = model_audio2vec
        self.feature_extractor = feature_extractor
        self.similarity_values = []

    def evaluate(self, prompt: str, description: str, target_audio: torch.Tensor, save_audio: bool = False):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        pred_audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(
            device).squeeze(0)
        pred_audio, target_audio = receive_audios(pred_audio, target_audio, False)
        inputs_pred = self.feature_extractor(pred_audio, return_tensors="pt").to(device)
        embeddings_pred = self.model_audio2vec(**inputs_pred).embeddings.to(device)
        inputs_target = self.feature_extractor(target_audio, return_tensors="pt")
        embeddings_target = self.model_audio2vec(**inputs_target).embeddings.to(device)
        metric = torch.nn.CosineSimilarity(dim=-1)
        similarity = float(metric(embeddings_pred, embeddings_target))
        self.similarity_values.append(similarity)
        return similarity

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.similarity_values)), self.similarity_values)
        plt.title('Similarity to original')
        plt.xlabel('tests')
        plt.ylabel('scores')
        plt.xticks(np.arange(len(self.similarity_values)))
        plt.grid()
        plt.show()

    def remove_history(self):
        self.similarity_values = []
