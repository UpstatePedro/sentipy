from typing import Union

import numpy as np

from sentipy.activations import tanh, linear


class Normaliser:

    def __init__(self, x_min: Union[float, np.float], x_max: Union[float, np.float]):
        """Normalise inputs to lie in the range [-1, +1] or denormalise to natural range

        :param x_min: minimum value in feature series
        :param x_max: maximum value in feature series
        """
        self.x_min = x_min
        self.x_max = x_max

    def normalise(self, x: Union[float, np.float]) -> Union[float, np.float]:
        """Normalise input x

        :param x: input value for normalisation
        :return: normalised value
        """
        return (2 * (x - self.x_min) / (self.x_max - self.x_min)) - 1

    def denormalise(self, x: Union[float, np.float]) -> Union[float, np.float]:
        """Denormalise input x

        :param x: input value for denormalisation
        :return: denormalised value
        """
        return 0.5 * (x + 1) * (self.x_max - self.x_min) + self.x_min


class Neuron:

    def _get_activation(self, func_name):
        return {
            'tansig': tanh,
            'linear': linear
        }.get(func_name)

    def __init__(self, weights: np.ndarray, bias: float, activation: str):
        self.weights = weights
        self.bias = bias
        self.activation_func = self._get_activation(activation)

    def forward(self, input_arr: np.ndarray) -> np.ndarray:
        """

        :param input_arr:
        :return:
        """
        activation_potential = self.calculate_potential(input_arr)
        return self.activation_func(activation_potential)

    def calculate_potential(self, input_arr):
        return np.dot(input_arr, self.weights) + self.bias


class FaparCalculator:

    def __init__(self):
        """Calculates FAPAR from Sentinel-2 imagery
        """
        self.norm_b3 = Normaliser(x_min=0., x_max=0.253061520471542)
        self.norm_b4 = Normaliser(x_min=0., x_max=0.290393577911328)
        self.norm_b5 = Normaliser(x_min=0., x_max=0.305398915248555)
        self.norm_b6 = Normaliser(x_min=0.006637972542253, x_max=0.608900395797889)
        self.norm_b7 = Normaliser(x_min=0.013972727018939, x_max=0.753827384322927)
        self.norm_b8a = Normaliser(x_min=0.026690138082061, x_max=0.782011770669178)
        self.norm_b11 = Normaliser(x_min=0.016388074192258, x_max=0.493761397883092)
        self.norm_b12 = Normaliser(x_min=0., x_max=0.493025984460231)
        self.norm_cos_view_zenith = Normaliser(x_min=0.918595400582046, x_max=0.99999999999139)
        self.norm_cos_sun_zenith = Normaliser(x_min=0.342022871159208, x_max=0.936206429175402)
        self.norm_cos_rel_azimuth = Normaliser(x_min=-0.999999982118044, x_max=0.999999998910077)
        self.norm_fapar = Normaliser(x_min=0.000153013463222, x_max=0.977135096979553)

        self.neuron_1 = Neuron(
            weights=[0.268714454733421, -0.205473108029835, 0.281765694196018, 1.33744341225598, 0.390319212938497,
                     -3.61271434220335, 0.222530960987244, 0.821790549667255, -0.093664567310731, 0.019290146147447,
                     0.037364446377188,
                     ],
            bias=-0.88706836404028,
            activation='tansig'
        )
        self.neuron_2 = Neuron(
            weights=[-0.248998054599707, -0.571461305473124, -0.369957603466673, 0.246031694650909, 0.332536215252841,
                     0.438269896208887, 0.81900055189045, -0.93493149905931, 0.082716247651866, -0.286978634108328,
                     -0.035890968351662],
            bias=0.320126471197199,
            activation='tansig'
        )
        self.neuron_3 = Neuron(
            weights=[-0.16406357531588, -0.126303285737763, -0.253670784366822, -0.321162835049381, 0.06708228797358,
                     2.02983228865526, -0.023141228827722, -0.553176625657559, 0.059285451897783, -0.034334454541432,
                     -0.031776704097009],
            bias=0.610523702500117,
            activation='tansig'
        )
        self.neuron_4 = Neuron(
            weights=[0.130240753003835, 0.236781035723321, 0.131811664093253, -0.250181799267664, -0.011364149953286,
                     -1.85757321463352, -0.146860751013916, 0.528008831372352, -0.046230769098303, -0.034509608392235,
                     0.031884395036004],
            bias=-0.379156190833946,
            activation='tansig'
        )
        self.neuron_5 = Neuron(
            weights=[-0.029929946166941, 0.795804414040809, 0.348025317624568, 0.943567007518504, -0.276341670431501,
                     -2.94659418014259, 0.2894830735075, 1.04400695044018, -0.000413031960419, 0.403331114840215,
                     0.068427130526696],
            bias=1.35302339669057,
            activation='tansig'
        )

        self.neuron_6 = Neuron(
            weights=[2.12603881106449, -0.632044932794919, 5.59899578720625, 1.77044414057897, -0.267879583604849],
            bias=-0.336431283973339,
            activation='linear'
        )

    def process_inputs(self, input_arr: np.ndarray) -> Union[float, np.float]:
        # TODO: make this dynamic

        ## Validate

        ## Normalise
        normalised_arr = self._normalise(input_arr)
        ## Compute output
        y_norm = self._compute(normalised_arr)
        ## Denormalise
        y = self.norm_fapar.denormalise(y_norm)
        return y

    def _normalise(self, input_arr: np.ndarray) -> np.ndarray:
        """

        :param input_arr:
        :return:
        """
        return np.array([
            self.norm_b3.normalise(input_arr[0]),
            self.norm_b4.normalise(input_arr[1]),
            self.norm_b5.normalise(input_arr[2]),
            self.norm_b6.normalise(input_arr[3]),
            self.norm_b7.normalise(input_arr[4]),
            self.norm_b8a.normalise(input_arr[5]),
            self.norm_b11.normalise(input_arr[6]),
            self.norm_b12.normalise(input_arr[7]),
            self.norm_cos_view_zenith.normalise(input_arr[8]),
            self.norm_cos_sun_zenith.normalise(input_arr[9]),
            self.norm_cos_rel_azimuth.normalise(input_arr[10]),
        ])

    def _compute(self, normalised_arr: np.ndarray) -> np.float:
        # Layer 1
        n1 = self.neuron_1.forward(normalised_arr)
        n2 = self.neuron_2.forward(normalised_arr)
        n3 = self.neuron_3.forward(normalised_arr)
        n4 = self.neuron_4.forward(normalised_arr)
        n5 = self.neuron_5.forward(normalised_arr)

        # Layer 2
        return self.neuron_6.forward(np.array([n1, n2, n3, n4, n5]))
