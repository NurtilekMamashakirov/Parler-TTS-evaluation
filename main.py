from evalution.evaluation_scripts import evaluate_objective_metrics, evaluate_similarity, evaluate_wer, evaluate_dnsmos
from argparse import ArgumentParser, Namespace


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluate", dest="mode", required=True,
                        help="Choose evaluation's type", type=str)
    parser.add_argument("-s", "--save_audio", dest="save", required=False,
                        help="True - saving audios / False - not saving audios", type=bool,
                        default=False)
    parser.add_argument("-n", "--number_of_tests", dest="tests_quantity", required=False, type=int,
                        default=5)
    return parser.parse_args()


def main(args: Namespace):
    match args.mode:
        case "wer":
            evaluate_wer(tests_quantity=args.tests_quantity, write_audios=args.save)
        case "dnsmos":
            evaluate_dnsmos(tests_quantity=args.tests_quantity, write_audios=args.save)
        case "obj":
            evaluate_objective_metrics(tests_quantity=args.tests_quantity, write_audios=args.save)
        case "sim":
            evaluate_similarity(tests_quantity=args.tests_quantity, write_audios=args.save)
        case "all":
            evaluate_wer(write_audios=args.save)
            evaluate_dnsmos(write_audios=args.save)
            evaluate_objective_metrics(tests_quantity=args.tests_quantity, write_audios=args.save)
            evaluate_similarity(tests_quantity=args.tests_quantity, write_audios=args.save)
        case _:
            exit("Unknown mode")


if __name__ == "__main__":
    args = parse_args()
    main(args)
