from sentipy.lib.preprocessing import Normaliser


def test_normaliser_minimum():
    """Confirm that input equal to x_min is normalised to -1"""
    normaliser = Normaliser(x_min=10, x_max=20)
    normalised_input = normaliser.normalise(10)
    expected_output = -1.
    assert normalised_input == expected_output


def test_normaliser_middle():
    """Confirm that input equal to midway between x_min & x_max is normalised to 0."""
    normaliser = Normaliser(x_min=10, x_max=20)
    normalised_input = normaliser.normalise(15)
    expected_output = 0.
    assert normalised_input == expected_output


def test_normaliser_maximum():
    """Confirm that input equal to x_max is normalised to +1"""
    normaliser = Normaliser(x_min=10, x_max=20)
    normalised_input = normaliser.normalise(20)
    expected_output = 1.
    assert normalised_input == expected_output


def test_normaliser_out_of_range():
    """Confirm that input above x_max is normalised to >1"""
    normaliser = Normaliser(x_min=10, x_max=20)
    normalised_input = normaliser.normalise(25)
    expected_output = 2.
    assert normalised_input == expected_output


def test_denormaliser_minimum():
    """Confirm that input equal to -1. is denormalised to x_min"""
    normaliser = Normaliser(x_min=10, x_max=20)
    denormalised_input = normaliser.denormalise(-1.)
    expected_output = 10
    assert denormalised_input == expected_output


def test_denormaliser_middle():
    """Confirm that input equal to 0 is denormalised to half way between x_min and x_max"""
    x_min = 10
    x_max = 20
    normaliser = Normaliser(x_min=x_min, x_max=x_max)
    normalised_input = normaliser.denormalise(0.)
    expected_output = (x_min + x_max) / 2.
    assert normalised_input == expected_output


def test_denormaliser_maximum():
    """Confirm that input equal to +1 is denormalised to x_max"""
    x_min = 10
    x_max = 20
    normaliser = Normaliser(x_min=x_min, x_max=x_max)
    normalised_input = normaliser.denormalise(1.)
    expected_output = x_max
    assert normalised_input == expected_output


def test_normaliser_out_of_range():
    """Confirm that input greater than +1 is denormalised to a value proportionately greater than x_max"""
    x_min = 10
    x_max = 20
    normaliser = Normaliser(x_min=x_min, x_max=x_max)
    input = 2.
    normalised_input = normaliser.denormalise(input)
    expected_output = x_max + (input - 1) * 0.5 * (x_max - x_min)
    assert normalised_input == expected_output
