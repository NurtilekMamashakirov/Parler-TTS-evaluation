from evaluation_scripts import evaluate_objective_metrics, evaluate_wer, evaluate_dnsmos

prompts = [
    "Welcome to our online store. We offer a wide selection of products at competitive prices. Place your order today and enjoy free delivery",
    "I believe I can fly. I believe I can touch the sky.",
    "The morning sun rises over the calm ocean, casting golden rays across the water. The sound of gentle waves creates a soothing.",
    "To turn on the device, press and hold the power button for three seconds. Once the light indicator appears, the device",
    "How many times did you rewrite that code? I think it is an excellent example of discipline"
]
describes = ["Jenny's voice is quite monotone and very clear in this recording, yet the sound is quite confined. "
             "She speaks very fastly."] * len(prompts)
evaluate_objective_metrics(tests_quantity=5)
evaluate_wer(prompts, describes)
evaluate_dnsmos(prompts, describes)
