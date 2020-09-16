Getting started
===============


Installation
------------

sentipy is hosted on pypy and can therefore be pip installed:

.. code-block:: bash

    $ pip install sentipy



Usage
-----

The intended responsibility of the **sentipy** project is to process pre-loaded Sentinel imagery into useful derivative
indices such as those included in the S2 Toolbox (Fcover, Fapar, Canopy water). This is likely to expand into other spectral
indices as the project grows.

We therefore do not involve ourselves in the loading of bands / pixels from imagery; our contract assumes that you will be
able to provide pixel values for each band (as required) in the form of numpy (masked) arrays, and we take care of the
processing logic.

Our preferred method for loading Sentinel imagery is to use the `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ package.

.. code-block:: python

    # Initialise whichever calculator you wish to use
    calc = Fapar()

    # We're assuming here that you've loaded all the band values into a list called `band_values`
    input_arr = np.array(band_values)

    # All you have to do is call the calculator's `.run()` method on the input array and you'll get an output array of
    # matching size (but not shape) with your calculated value(s)
    output_fapar = calc.run(input_arr)

And that's really all there is to it!