lss2 - Large-Scale Structure
============================

`lss2` is a python module for computations related to the large-scale structure of the universe! The main tool is the `CosmoStructure` object. It can be used for computing the linear power spectra, variance or the halo mass function under some specified cosmology. Also it contains implementations of some transfer function, mass function and bias models.

Basic Usage
-----------

To create a `CosmoStructure` object for a specific cosmology model. 

```{python}
>>> import lss2
>>> lss2.CosmoStrucure()
```
