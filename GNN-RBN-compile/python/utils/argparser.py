import argparse


def argument_parser():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results
    # reported in the paper.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train', type=str,
                        help='data to train')
    parser.add_argument('--data_test', type=str,
                        help='data to test')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed (default: 0)')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=3,
        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument(
        '--num_mlp_layers',
        type=int,
        default=2,
        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument(
        '--combine_layers',
        type=int,
        default=1,
        help='number of layers for MLP combne EXCLUDING the input one (default: 2).')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=3,
        help='number of hidden units for node representation (default: 64)')
    parser.add_argument(
        '--readout',
        type=str,
        default="mean",
        choices=[
            "add",
            "mean",
            "max"],
        help='Pooling for over all nodes in a graph: add, mean or max')
    parser.add_argument(
        '--aggregate',
        type=str,
        default="add",
        choices=[
            "add",
            "mean",
            "max"],
        help='Pooling for over neighboring nodes: add, mean or max')
    parser.add_argument('--filename', type=str, default="training.log",
                        help='output file')
    parser.add_argument(
        '--task_type',
        type=str,
        default="graph",
        choices=[
            "node",
            "graph"],
        help='Task to solve, `node` or `graph` classification')
    return parser