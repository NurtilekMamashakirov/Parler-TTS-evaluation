from evaluation_scripts import evaluate_objective_metrics, evaluate_wer, evaluate_dnsmos

prompts = [
    "Hello, my name is Jenny. How are you? I drunk a lot of martini and can't walk by myself anymore!",
    "I believe I can fly. I believe I can touch the sky.",
    "How many times did you rewrite that code? I think it is an excellent example of discipline",
    "There is so many lies about me. I think it is false",
    "Do you think Cristiano can win the World Cup in twenty twenty six in the age of fourty"
]
describes = ["Jenny's voice is quite monotone and very clear in this recording, yet the sound is quite confined. "
             "She speaks very slowly."] * len(prompts)
evaluate_objective_metrics(tests_quantity=5)
evaluate_wer(prompts, describes)
evaluate_dnsmos(prompts, describes)
