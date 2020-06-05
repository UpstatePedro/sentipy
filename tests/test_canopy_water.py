"""No test cases available in the ESA S2 toolbox excel"""
import numpy as np
import pytest

from sentipy.s2_toolbox import CanopyWater


def test_validation_with_validate_flag():
    calc = CanopyWater()
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
        canopy_water = calc.run(input_arr, validate=True)

def test_validation_without_validate_flag():
    calc = CanopyWater()
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
    canopy_water = calc.run(input_arr, validate=False)
    assert type(canopy_water) == np.float64
