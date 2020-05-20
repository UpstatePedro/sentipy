import pytest

from sentipy.fapar import FaparCalculator
import numpy as np


def test_valid_inputs_1():
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

    calc = FaparCalculator()
    input_arr = np.array([B3, B4, B5, B6, B7, B8A, B11, B12, View_Zenith, Sun_Zenith, Rel_Azimuth])
    fapar = calc.run(input_arr)
    np.testing.assert_almost_equal(fapar, expected_FAPAR)

def test_valid_inputs_1_with_different_sequence():
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

    calc = FaparCalculator()
    band_sequence = [
        "COS_REL_AZIMUTH", "B03", "B04", "B05", "B06", "B07", "B8a", "B11", "B12", "COS_VIEW_ZENITH", "COS_SUN_ZENITH"
    ]
    input_arr = np.array([Rel_Azimuth, B3, B4, B5, B6, B7, B8A, B11, B12, View_Zenith, Sun_Zenith])
    fapar = calc.run(input_arr, band_sequence)
    np.testing.assert_almost_equal(fapar, expected_FAPAR)


def test_normalise_min():
    calc = FaparCalculator()
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
    calc = FaparCalculator()
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
    calc = FaparCalculator()
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
    calc = FaparCalculator()
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
