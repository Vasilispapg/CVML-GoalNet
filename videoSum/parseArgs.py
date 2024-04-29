from argparse import ArgumentParser
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Infer the model')
    return parser.parse_args()