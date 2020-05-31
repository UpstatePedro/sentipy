import pytest

from sentipy.s2_toolbox import Fapar
import numpy as np


validation_examples = [
    ([0.066937, 0.14317, 0.1639, 0.17261, 0.19064, 0.20866, 0.25406, 0.20098, 0.97314, 0.87393, 0.86514], 0.05234725),
    ([0.057861, 0.0031582, 0.071919, 0.31941, 0.45102, 0.41836, 0.12667, 0.052655, 0.93696, 0.71347, -0.6401], 0.8850626),
    ([0.10733, 0.07007, 0.1355, 0.2912, 0.33657, 0.3335, 0.16552, 0.11706, 0.9694, 0.86803, -0.82835], 0.574918),
    ([0.093999, 0.038522, 0.10442, 0.40728, 0.49859, 0.55073, 0.16706, 0.053842, 0.96141, 0.86044, -0.95461], 0.923845),
    ([0.12239, 0.084862, 0.13476, 0.2043, 0.24815, 0.25077, 0.1696, 0.11165, 0.99984, 0.82341, 0.82177], 0.3937618),
    ([0.21207, 0.2414, 0.23862, 0.27521, 0.33035, 0.33301, 0.34619, 0.32453, 0.99653, 0.847, 0.98009], 0.1027861),
    ([0.062781, 0.044145, 0.05917, 0.16264, 0.19849, 0.20459, 0.14473, 0.11724, 0.99171, 0.40177, 0.57041], 0.5441121),
]

@pytest.mark.parametrize("band_values, expected_fapar", validation_examples)
def test_valid_inputs(band_values, expected_fapar):
    calc = Fapar()
    input_arr = np.array(band_values)
    output_fapar = calc.run(input_arr)
    np.testing.assert_almost_equal(output_fapar, expected_fapar, decimal=5)

def test_valid_inputs_with_different_sequence():
    B3 = 0.066937
    B4 = 0.14317
    B5 = 0.1639
    B6 = 0.17261
    B7 = 0.19064
    B8A = 0.20866
    B11 = 0.25406
    B12 = 0.20098
    View_Zenith = 0.97314
    Sun_Zenith = 0.87393
    Rel_Azimuth = 0.86514
    expected_FAPAR = 0.05234725

    calc = Fapar()
    band_sequence = [
        "COS_REL_AZIMUTH", "B03", "B04", "B05", "B06", "B07", "B8a", "B11", "B12", "COS_VIEW_ZENITH", "COS_SUN_ZENITH"
    ]
    input_arr = np.array([Rel_Azimuth, B3, B4, B5, B6, B7, B8A, B11, B12, View_Zenith, Sun_Zenith])
    fapar = calc.run(input_arr, band_sequence)
    np.testing.assert_almost_equal(fapar, expected_FAPAR)


def test_normalise_min():
    calc = Fapar()
    B3_min = calc.norm_b3.x_min
    B4_min = calc.norm_b4.x_min
    B5_min = calc.norm_b5.x_min
    B6_min = calc.norm_b6.x_min
    B7_min = calc.norm_b7.x_min
    B8A_min = calc.norm_b8a.x_min
    B11_min = calc.norm_b11.x_min
    B12_min = calc.norm_b12.x_min
    View_Zenith_min = calc.norm_cos_view_zenith.x_min
    Sun_Zenith_min = calc.norm_cos_sun_zenith.x_min
    Rel_Azimuth_min = calc.norm_cos_rel_azimuth.x_min

    output = calc._normalise(np.array(
        [B3_min, B4_min, B5_min, B6_min, B7_min, B8A_min, B11_min, B12_min, View_Zenith_min, Sun_Zenith_min,
         Rel_Azimuth_min])
    )
    expected_output = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
    np.testing.assert_equal(output, expected_output)


def test_normalise_max():
    calc = Fapar()
    B3_min = calc.norm_b3.x_max
    B4_min = calc.norm_b4.x_max
    B5_min = calc.norm_b5.x_max
    B6_min = calc.norm_b6.x_max
    B7_min = calc.norm_b7.x_max
    B8A_min = calc.norm_b8a.x_max
    B11_min = calc.norm_b11.x_max
    B12_min = calc.norm_b12.x_max
    View_Zenith_min = calc.norm_cos_view_zenith.x_max
    Sun_Zenith_min = calc.norm_cos_sun_zenith.x_max
    Rel_Azimuth_min = calc.norm_cos_rel_azimuth.x_max

    output = calc._normalise(np.array(
        [B3_min, B4_min, B5_min, B6_min, B7_min, B8A_min, B11_min, B12_min, View_Zenith_min, Sun_Zenith_min,
         Rel_Azimuth_min])
    )
    expected_output = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    np.testing.assert_equal(output, expected_output)


def test_validation_with_validate_flag():
    calc = Fapar()
    # valid_inputs
    B3 = 0.066937
    B4 = 0.14317
    B5 = 0.1639
    B6 = 0.17261
    B7 = 0.19064
    B8A = 0.20866
    B11 = 0.25406
    B12 = 0.20098
    View_Zenith = 0.97314
    Sun_Zenith = 0.87393
    Rel_Azimuth = 0.86514

    # Overwrite with invalid
    B7 = 10.2
    input_arr = np.array([B3, B4, B5, B6, B7, B8A, B11, B12, View_Zenith, Sun_Zenith, Rel_Azimuth])

    # Run the calculator and assert exceptions raised
    with pytest.raises(ValueError, match=r".* B07 .* [0.0, 1.0]"):
        fapar = calc.run(input_arr, validate=True)


def test_validation_without_validate_flag():
    calc = Fapar()
    # valid_inputs
    B3 = 0.066937
    B4 = 0.14317
    B5 = 0.1639
    B6 = 0.17261
    B7 = 0.19064
    B8A = 0.20866
    B11 = 0.25406
    B12 = 0.20098
    View_Zenith = 0.97314
    Sun_Zenith = 0.87393
    Rel_Azimuth = 0.86514

    # Overwrite with invalid
    B7 = 10.2
    input_arr = np.array([B3, B4, B5, B6, B7, B8A, B11, B12, View_Zenith, Sun_Zenith, Rel_Azimuth])

    # Run the calculator and confirming that no exceptions are raised by asserting we get a results
    fapar = calc.run(input_arr, validate=False)
    assert type(fapar) == np.float64
