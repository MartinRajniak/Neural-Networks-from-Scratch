from src.model.variable import Variable
from src.model.neuron import neuron

def build_model(n_inputs, n_hidden_layers, n_hidden_neurons):
    inputs = [Variable(f"input{input_index}") for input_index in range(n_inputs)]

    layers_inputs = inputs
    for _ in range(n_hidden_layers):
        hidden_neurons = []
        for _ in range(n_hidden_neurons):
            hidden_neurons.append(neuron(layers_inputs, use_activation=True))
        layers_inputs = hidden_neurons

    output = neuron(layers_inputs)

    target = Variable("target")
    loss = output.mse(target)
    loss.label = "loss"

    return inputs, target, output, loss
