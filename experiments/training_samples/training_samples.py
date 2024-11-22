import sys
import os
import click

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import *

from src.loss import cross_entropy_loss

def test_n_samples_function(model_name: str, n_samples, max_len: int = 10, n_epochs: int = 50000, batch_size: int = 256, learning_rate: float = 1e-04, output_dir: str = None):
    model = generateModel(model_name, max_len)
    model.setRandomWeights()
    X, Y = generateAndEncodeData(model_name, max_len, n_samples, removeDuplicates=True)

    X_train = X[:int(0.85*len(X))]
    Y_train = Y[:int(0.85*len(Y))]
    X_val = X[int(0.85*len(X)):]
    Y_val = Y[int(0.85*len(Y)):]

    return model.train(X_train,Y_train, n_epochs, batch_size, learning_rate, False, X_val, Y_val, 0, 10, loss_fn=cross_entropy_loss, output_dir=output_dir)

@click.command()
@click.option("--model_name", type=str, help="The name of the program/model to train")
@click.option("--n_samples", type=int, help="The number of samples to generate for training/validation")
def run_test(model_name, n_samples):
    if model_name not in getAcceptedNamesAndInput().keys():
        raise ValueError(f"Model {model_name} not found")

    print(f"Testing {model_name} with {n_samples} samples...")
    metrics, validation = test_n_samples_function(model_name, n_samples, output_dir=f"saved_data/{model_name}/{n_samples}/")
    return model_name, n_samples, metrics, validation

if __name__ == "__main__":
    run_test()