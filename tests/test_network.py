from sentipy.lib.neuralnet import Network, Neuron
import numpy as np

def test_network_initialisation():
    neuron1 = Neuron(
        weights=np.array([1., 1., 1., 1., 1.]),
        bias=0.0,
        activation='tansig'
    )

    neuron2 = Neuron(
        weights=np.array([1., 1., 1., 1., 1.]),
        bias=0.0,
        activation='tansig'
    )

    neuron3 = Neuron(
        weights=np.array([1., 1.]),
        bias=0.0,
        activation='linear'
    )
    network = Network(
        hidden_layers=(
            [neuron1, neuron2],
        ),
        output_neuron=neuron3
    )
    inputs = np.array([0., 0., 0., 0., 0.])
    output = network.forward(inputs)
    expected_output = 0
    assert output == expected_output
