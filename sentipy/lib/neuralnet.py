from typing import Tuple, List

import numpy as np

from sentipy.lib import activations


class Neuron:

    def _get_activation(self, func_name):
        return {
            'tansig': activations.tanh,
            'linear': activations.linear
        }.get(func_name)

    def __init__(self, weights: np.ndarray, bias: float, activation: str):
        self.weights = weights
        self.bias = bias
        self.activation_func = self._get_activation(activation)

    def forward(self, input_arr: np.ndarray) -> np.float:
        """

        :param input_arr:
        :return:
        """
        activation_potential = self.calculate_potential(input_arr)
        return self.activation_func(activation_potential)

    def calculate_potential(self, input_arr):
        return np.dot(input_arr, self.weights) + self.bias


class Network:

    def __init__(self, hidden_layers: Tuple[List[Neuron]], output_neuron: Neuron):
        self._hidden_layers = hidden_layers
        self._output_neuron = output_neuron

    def forward(self, input_arr: np.ndarray) -> np.float:
        layer_inputs = input_arr
        for layer in self._hidden_layers:
            layer_outputs = np.array([neuron.forward(layer_inputs) for neuron in layer])
            layer_inputs = layer_outputs
        return self._output_neuron.forward(layer_outputs)