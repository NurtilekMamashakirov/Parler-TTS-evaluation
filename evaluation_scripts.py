import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from evaluators import DNSMOSEvaluator, WEREvaluator, ObjectiveMetricsEvaluator
from utils import generate_prompts_with_gigachat, generate_describes_with_gigachat, AudioDataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_tts = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")


def evaluate_dnsmos(prompts=None, describes=None):
    dnsmos_evaluator = DNSMOSEvaluator(model_tts, tokenizer)
    if prompts is None:
        prompts = generate_prompts_with_gigachat()
    if describes is None:
        describes = generate_describes_with_gigachat()
    for prompt, describe in zip(prompts, describes):
        dnsmos_tensor = dnsmos_evaluator.evaluate(prompt, describe)
        print("DNSMOS evaluation")
        print(f'Overall MOS: {dnsmos_tensor[0]}')
        print(f'Signal Qualities: {dnsmos_tensor[1]}')
        print(f'Background Noise: {dnsmos_tensor[2]}')
        print(f'Linguistic: {dnsmos_tensor[3]}\n')
    dnsmos_evaluator.visualize()


def evaluate_wer(prompts=None, describes=None):
    model_stt = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True, use_safetensors=True)
    model_stt.to(device)
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_stt,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    if prompts is None:
        prompts_quantity = len(describes) if describes else 5
        prompts = generate_prompts_with_gigachat(prompts_quantity)
    if describes is None:
        describes_quantity = len(prompts) if prompts else 5
        describes = generate_describes_with_gigachat(describes_quantity)

    wer_eval = WEREvaluator(model_tts, tokenizer, pipe)
    for prompt, describe in zip(prompts, describes):
        print(f"Word Error rate: {wer_eval.evaluate(prompt, describe)}")
    wer_eval.visualize()


def evaluate_objective_metrics(tests_quantity: int, random_seed: int = None):
    dataset = AudioDataset(tests_quantity, random_seed=random_seed)
    objective_metrics_evaluator = ObjectiveMetricsEvaluator(model_tts, tokenizer)
    for i in range(len(dataset)):
        features, target_audio = dataset[i]
        objective_metrics = objective_metrics_evaluator.evaluate(features["transcription_normalised"],
                                                                 features["text_description"],
                                                                 torch.tensor(target_audio))
        print("Objective metrics:")
        print(f"pesq: {objective_metrics['pesq']}")
        print(f"si-sdr: {objective_metrics['si_sdr']}")
        print(f"stoi: {objective_metrics['stoi']}\n")
    objective_metrics_evaluator.visualize()
