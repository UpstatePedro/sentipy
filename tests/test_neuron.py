from sentipy import activations
from sentipy.fapar import Neuron
import numpy as np

def test_neuron_tansig_activation():
    neuron = Neuron(
        weights=[0.1, 0.2, 0.3, 0.4, 0.5],
        bias=0.1,
        activation='tansig'
    )
    assert neuron.activation_func.__name__ == activations.tanh.__name__

def test_neuron_linear_activation():
    neuron = Neuron(
        weights=[0.1, 0.2, 0.3, 0.4, 0.5],
        bias=0.1,
        activation='linear'
    )
    assert neuron.activation_func.__name__ == activations.linear.__name__

def test_calculate_potential():
    neuron = Neuron(
        weights=[0.5, 0.5, 0.5],
        bias=0.5,
        activation='tansig'
    )
    input_arr = np.array([1., 1., 1.])

    output = neuron.calculate_potential(input_arr)
    expected_output = 3. * 0.5 + 0.5
    assert output == expected_output

def test_tansig_activation():
    x = 1.
    output = activations.tanh(x)
    expected_output = 2. / (1 + np.exp(-2. * x)) - 1
    np.testing.assert_almost_equal(output, expected_output)
