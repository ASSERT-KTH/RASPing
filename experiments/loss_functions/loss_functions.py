import sys
import os
import click

# TODO: this is a hack, change this for editable install
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.functions import *

from src.loss import cross_entropy_loss, cross_entropy_loss_smoothed_accuracy, cross_entropy_loss_with_perfect_sequence

def test_loss_function(model_name: str, loss_fn, max_len: int = 10, n_samples: int = 5000, n_epochs: int = 50000, batch_size: int = 256, learning_rate: float = 1e-04, output_dir: str = None):
    model = generateModel(model_name, max_len)
    X, Y = generateAndEncodeData(model_name, max_len, n_samples, removeDuplicates=True)

    X_train = X[:int(0.85*len(X))]
    Y_train = Y[:int(0.85*len(Y))]
    X_val = X[int(0.85*len(X)):]
    Y_val = Y[int(0.85*len(Y)):]

    return model.train(X_train,Y_train, n_epochs, batch_size, learning_rate, False, X_val, Y_val, 0, 10, loss_fn=loss_fn, output_dir=output_dir)

@click.command()
@click.option("--model_name", type=str, help="The name of the program/model to train")
@click.option("--loss_fn_name", type=str, help="The name of the loss function to test")
def run_test(model_name, loss_fn_name):
    if model_name not in getAcceptedNamesAndInput().keys():
        raise ValueError(f"Model {model_name} not found")

    loss_fns = {
        "cross_entropy_loss": cross_entropy_loss,
        "cross_entropy_loss_smoothed_accuracy": cross_entropy_loss_smoothed_accuracy,
        "cross_entropy_loss_with_perfect_sequence": cross_entropy_loss_with_perfect_sequence
    }
    if loss_fn_name not in loss_fns:
        raise ValueError(f"Loss function {loss_fn_name} not found")
    loss_fn = loss_fns[loss_fn_name]

    print(f"Testing {model_name} with {loss_fn_name}...")
    metrics, validation = test_loss_function(model_name, loss_fn, output_dir=f"saved_data/{model_name}/{loss_fn_name}/")
    return model_name, loss_fn_name, metrics, validation

if __name__ == "__main__":
    run_test()