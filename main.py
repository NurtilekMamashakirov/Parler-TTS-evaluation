from evalution.evaluation_scripts import evaluate_objective_metrics, evaluate_similarity, evaluate_wer, evaluate_dnsmos
from argparse import ArgumentParser, Namespace


# fixed_prompts = [
#     "Welcome to our online store. We offer a wide selection of products at competitive prices. Place your order "
#     "today and enjoy free delivery",
#     "I believe I can fly. I believe I can touch the sky.",
#     "The morning sun rises over the calm ocean, casting golden rays across the water. The sound of gentle waves "
#     "creates a soothing.",
#     "To turn on the device, press and hold the power button for three seconds. Once the light indicator appears, "
#     "the device",
#     "How many times did you rewrite that code? I think it is an excellent example of discipline"
# ]
# fixed_describes = ["Jenny's voice is quite monotone and very clear in this recording. She speaks very fast."] * len(
#     fixed_prompts)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluate", dest="mode", required=True,
                        help="Choose evaluation's type", type=str)
    parser.add_argument("-s", "--save_audio", dest="save", required=False,
                        help="True - saving audios / False - not saving audios", type=bool,
                        default=False)
    return parser.parse_args()


def main(args: Namespace):
    match args.mode:
        case "wer":
            evaluate_wer(write_audios=args.save)
        case "dnsmos":
            evaluate_dnsmos(write_audios=args.save)
        case "obj":
            evaluate_objective_metrics(tests_quantity=5, write_audios=args.save)
        case "sim":
            evaluate_similarity(tests_quantity=5, write_audios=args.save)
        case "all":
            evaluate_wer(write_audios=args.save)
            evaluate_dnsmos(write_audios=args.save)
            evaluate_objective_metrics(tests_quantity=5, write_audios=args.save)
            evaluate_similarity(tests_quantity=5, write_audios=args.save)
        case _:
            exit("Unknown mode")


if __name__ == "__main__":
    args = parse_args()
    main(args)
