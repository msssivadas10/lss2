lss2 - Large-Scale Structure
============================

`lss2` is a python module for computations related to the large-scale structure of the universe! The main tool is the `CosmoStructure` object. It can be used for computing the linear power spectra, variance or the halo mass function under some specified cosmology. Also it contains implementations of some transfer function, mass function and bias models.

Installation
------------

The module can be installed by running the `setup.py` script.

```
$ python3 setup.py install
```

This module depends on numpy and scipy packages. Tests were done with Python3, on an Ubuntu 18 machine

Basic Usage
-----------

To create a `CosmoStructure` object for a specific cosmology model with default settings, 

```{python}
>>> import lss2
>>> cs = lss2.CosmoStructure(Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, n = 1., h = 0.7)
```
This can be then used to get the power spectrum or mass-function with this this cosmology!
