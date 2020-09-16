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

    from sentipy import s2_toolbox
    import rasterio

    with rasterio.open('example.tif') as dataset:
        # read your bands in & stack them into a single array here
        input_arr = np.stack(
            [
                dataset.read(1),
                dataset.read(2),
                dataset.read(3),
                # ...
                # and you'll need the image meta-data here too: view zenith, sun zenith, rel azimuth
            ]
            , axis=0
        )

    # Initialise whichever calculator you wish to use, eg the Fapar calculator:
    calculator = s2_toolbox.Fapar()
    # All you have to do is call the calculator's `.run()` method on the input array and you'll get an output array of
    # matching size (but not shape) with your calculated value(s)
    fapar = calculator.run(input_arr)

And that's really all there is to it!