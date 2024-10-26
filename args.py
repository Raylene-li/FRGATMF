"""Parsing the model parameters."""

import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Twitch Brasilians dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node identifiers.
    """
    parser = argparse.ArgumentParser(description="Run FRGATMF.")

    parser.add_argument("--num_node",
                        type=int,
                        default=251,
                        help="Acupuncture: 120, winconsin: 251")

    parser.add_argument("--hid_features",
                        type=int,
                        default=128,
                        help="generator hid_features.")

    parser.add_argument("--sec_hidden",
                        type=int,
                        default=64,
                        help="second hidden layer.")

    parser.add_argument("--nb_heads",
                        type=int,
                        default=2,
                        help="Number of gat head attentions.")

    parser.add_argument("--gat_patience",
                        type=int,
                        default=20,
                        help="Number of gat hidden units.")

    parser.add_argument("--gat_alpha",
                        type=float,
                        default=0.2,
                        help="Alpha for the leaky_relu.")

    parser.add_argument("--gat_lr",
                        type=float,
                        default=0.001,
                        help="Initial gat learning rate.")

    parser.add_argument("--gat_weight_decay",
                        type=float,
                        default=1e-4,
                        help="Initial gat learning rate.")

    parser.add_argument("--gat_epoches",
                        type=int,
                        default=100,
                        help="gat_epoches.")

    parser.add_argument("--iterations",
                        type=int,
                        default=100,
	                help="Number of training iterations. Default is 100.")

    parser.add_argument("--pre-iterations",
                        type=int,
                        default=100,
	                help="Number of layerwsie pre-training iterations. Default is 100.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=0.1,
	                help="Regularization parameter. Default is 0.01.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 128 64 32.")

    parser.add_argument("--calculate-loss",
                        dest="calculate_loss",
                        action="store_true")

    parser.add_argument("--not-calculate-loss",
                        dest="calculate_loss",
                        action="store_false")

    parser.add_argument("--threshold",
                        type=float,
                        default=0.7,
                        help="Acupuncture: 0.7, Cora:0.3, citeseer and winconsin: 0.4")

    parser.set_defaults(calculate_loss=True)

    parser.set_defaults(layers=[64, 32])  # 120, 64   64, 32

    return parser.parse_args()
