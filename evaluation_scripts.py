import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2FeatureExtractor, \
    WavLMForXVector

from evaluators import DNSMOSEvaluator, WEREvaluator, ObjectiveMetricsEvaluator, SimilarityEvaluator
from utils import generate_prompts_with_gigachat, generate_describes_with_gigachat, prepare_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_tts = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")


def evaluate_dnsmos(prompts: list = None, describes: list = None, write_audios=False):
    dnsmos_evaluator = DNSMOSEvaluator(model_tts, tokenizer)
    if prompts is None:
        prompts_quantity = len(describes) if describes else 5
        prompts = generate_prompts_with_gigachat(prompts_quantity)
    if describes is None:
        describes_quantity = len(prompts) if prompts else 5
        describes = generate_describes_with_gigachat(describes_quantity)
    for prompt, describe in zip(prompts, describes):
        dnsmos_tensor = dnsmos_evaluator.evaluate(prompt, describe, write_audios)
        print("DNSMOS evaluation")
        print(f'Overall MOS: {dnsmos_tensor[0]}')
        print(f'Signal Qualities: {dnsmos_tensor[1]}')
        print(f'Background Noise: {dnsmos_tensor[2]}')
        print(f'Linguistic: {dnsmos_tensor[3]}\n')
    dnsmos_evaluator.visualize()


def evaluate_wer(prompts: list = None, describes: list = None, write_audios: bool = False):
    model_stt = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True, use_safetensors=True)
    model_stt.to(device)
    model_stt.generation_config.language = "en"
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
        print(f"Word Error rate: {wer_eval.evaluate(prompt, describe, write_audios)}")
    wer_eval.visualize()


def evaluate_objective_metrics(tests_quantity: int, random_seed: int = None, write_audios: bool = False):
    dataset = prepare_dataset(tests_quantity, random_seed)
    objective_metrics_evaluator = ObjectiveMetricsEvaluator(model_tts, tokenizer)
    for i in range(tests_quantity):
        objective_metrics = objective_metrics_evaluator.evaluate(dataset["transcription"][i],
                                                                 dataset["text_description"][i],
                                                                 torch.tensor(dataset["audio"][i]["array"]),
                                                                 write_audios)
        print("Objective metrics:")
        print(f"pesq: {objective_metrics['pesq']}")
        print(f"si-sdr: {objective_metrics['si_sdr']}")
        print(f"stoi: {objective_metrics['stoi']}\n")
    objective_metrics_evaluator.visualize()


def evaluate_similarity(tests_quantity: int, random_seed: int = None, write_audios: bool = False):
    dataset = prepare_dataset(tests_quantity, random_seed)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
    similarity_evaluator = SimilarityEvaluator(model_tts, tokenizer, model, feature_extractor)
    for i in range(tests_quantity):
        similarity = similarity_evaluator.evaluate(dataset["transcription"][i],
                                                   dataset["text_description"][i],
                                                   torch.tensor(dataset["audio"][i]["array"]),
                                                   write_audios)
        print(f"Similarity metrics: {similarity}")
    similarity_evaluator.visualize()
