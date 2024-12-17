import numpy as np
import torch
from jiwer import wer
from matplotlib import pyplot as plt
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility, \
    ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DNSMOSEvaluator:
    def __init__(self, model_tts, tokenizer):
        super().__init__()
        self.model_tts = model_tts.to(
            device)
        self.tokenizer = tokenizer
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=self.model_tts.config.sampling_rate, personalized=False)
        self.overall_moses = []
        self.signal_qualities = []
        self.background_noises = []
        self.linguistic_artifacts = []

    def evaluate(self, prompt: str, description: str):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(device)
        metric = self.dnsmos(audio)
        self.overall_moses.append(metric[0])
        self.signal_qualities.append(metric[1])
        self.background_noises.append(metric[2])
        self.linguistic_artifacts.append(metric[3])
        return self.dnsmos(audio)

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.overall_moses)), self.overall_moses, label='Overall MOS')
        plt.scatter(np.arange(len(self.overall_moses)), self.signal_qualities, label='Signal Qualities')
        plt.scatter(np.arange(len(self.overall_moses)), self.background_noises, label='Background Noise')
        plt.scatter(np.arange(len(self.overall_moses)), self.linguistic_artifacts, label='Linguistic')
        plt.title('DNSMOS Evaluation')
        plt.xlabel('tests')
        plt.ylabel('scores')
        plt.xticks(np.arange(len(self.overall_moses)))
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def remove_history(self):
        self.overall_moses = []
        self.signal_qualities = []
        self.background_noises = []
        self.linguistic_artifacts = []


class WEREvaluator:
    def __init__(self, model_tts, tokenizer, pipeline_stt):
        self.model_tts = model_tts.to(device)
        self.tokenizer = tokenizer
        self.pipeline_stt = pipeline_stt
        self.word_error_rates = []

    def evaluate(self, prompt: str, description: str):
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

    def evaluate(self, prompt: str, description: str, target_audio: torch.Tensor):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        pred_audio = self.model_tts.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(
            device).squeeze(0)
        target_audio = target_audio.to(device)
        if len(pred_audio) > len(target_audio):
            dif = len(pred_audio) - len(target_audio)
            target_audio = torch.cat((target_audio, torch.zeros(dif)), dim=0)
        if len(pred_audio) < len(target_audio):
            dif = len(target_audio) - len(pred_audio)
            pred_audio = torch.cat((pred_audio, torch.zeros(dif)), dim=0)
        pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode="nb")
        pesq = pesq_metric(pred_audio, target_audio)
        self.perceptual_evaluation_of_speech_qualities.append(float(pesq))

        si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
        si_sdr = si_sdr_metric(pred_audio, target_audio)
        self.scale_invariant_signal_to_distortion_ratios.append(float(si_sdr))

        stoi_metric = ShortTimeObjectiveIntelligibility(fs=self.model_tts.config.sampling_rate, extended=False).to(device)
        stoi = stoi_metric(pred_audio, target_audio)
        self.short_time_objective_intelligibility.append(float(stoi))
        return {'pesq': pesq, 'si_sdr': si_sdr, 'stoi': stoi}

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.perceptual_evaluation_of_speech_qualities)),
                    self.perceptual_evaluation_of_speech_qualities, label='PESQ')
        average_pesq = np.array(self.perceptual_evaluation_of_speech_qualities).mean()
        plt.plot(np.arange(len(self.perceptual_evaluation_of_speech_qualities)),
                 len(self.perceptual_evaluation_of_speech_qualities) * [average_pesq],
                 label=f'Average score = {average_pesq:.2f}')
        plt.title('PESQ metric')
        plt.xlabel('tests')
        plt.ylabel('PESQ score')
        plt.xticks(np.arange(len(self.perceptual_evaluation_of_speech_qualities)))
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.scale_invariant_signal_to_distortion_ratios)),
                    self.scale_invariant_signal_to_distortion_ratios, label='SI-SDR')
        average_si_sdr = np.array(self.scale_invariant_signal_to_distortion_ratios).mean()
        plt.plot(np.arange(len(self.scale_invariant_signal_to_distortion_ratios)),
                 len(self.scale_invariant_signal_to_distortion_ratios) * [average_si_sdr],
                 label=f'Average score = {average_si_sdr:.2f}')
        plt.title('SI-SDR metric')
        plt.xlabel('tests')
        plt.ylabel('SI-SDR scores')
        plt.xticks(np.arange(len(self.perceptual_evaluation_of_speech_qualities)))
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.figure(figsize=(12, 6))
        plt.scatter(np.arange(len(self.short_time_objective_intelligibility)),
                    self.short_time_objective_intelligibility, label='STOI')
        average_stoi = np.array(self.short_time_objective_intelligibility).mean()
        plt.plot(np.arange(len(self.short_time_objective_intelligibility)),
                 len(self.short_time_objective_intelligibility) * [average_stoi],
                 label=f'Average score = {average_stoi:.2f}')
        plt.title('STOI metric')
        plt.xlabel('tests')
        plt.ylabel('STOI score')
        plt.xticks(np.arange(len(self.perceptual_evaluation_of_speech_qualities)))
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
