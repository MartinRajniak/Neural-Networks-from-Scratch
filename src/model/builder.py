from src.model.variable import Variable
from src.model.neuron import neuron

def build_model(n_inputs, n_hidden_layers, n_hidden_neurons, output_activation, loss_fn):
    inputs = [Variable(f"input{input_index}") for input_index in range(n_inputs)]

    layers_inputs = inputs
    for _ in range(n_hidden_layers):
        hidden_neurons = [
            neuron(layers_inputs, activation="tanh" if loss_fn == "mse" else "relu")
            for _ in range(n_hidden_neurons)
        ]
        layers_inputs = hidden_neurons

    output = neuron(layers_inputs, activation=output_activation) if output_activation else neuron(layers_inputs)

    target = Variable("target")
    if loss_fn == "mse":
        loss = output.mse(target)
    elif loss_fn == "bce":
        loss = output.bce(target)
    else:
        raise ValueError("Unsupported loss function")
    loss.label = "loss"

    return inputs, target, output, loss

def build_regression_model(n_inputs, n_hidden_layers, n_hidden_neurons):
    return build_model(n_inputs, n_hidden_layers, n_hidden_neurons, output_activation=None, loss_fn="mse")

def build_classification_model(n_inputs, n_hidden_layers, n_hidden_neurons):
    return build_model(n_inputs, n_hidden_layers, n_hidden_neurons, output_activation="sigmoid", loss_fn="bce")
