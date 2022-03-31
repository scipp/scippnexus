scipp - Multi-dimensional data arrays with labeled dimensions
=============================================================

.. raw:: html

   <span style="font-size:1.2em;font-style:italic;color:#5a5a5a">
      A Python library enabling a modern and intuitive way of working with scientific data in Jupyter notebooks
   </span>

**scipp** is heavily inspired by `xarray <https://xarray.pydata.org>`_.
It enriches raw NumPy-like multi-dimensional arrays of data by adding named dimensions and associated coordinates.
Multiple arrays can be combined into datasets.
While for many applications xarray is certainly more suitable (and definitely much more matured) than scipp, there is a number of features missing in other situations.
If your use case requires one or several of the items on the following list, using scipp may be worth considering:

- **Physical units** are stored with each data or coord array and are handled in arithmetic operations.
- **Propagation of uncertainties**.
- Support for **histograms**, i.e., **bin-edge axes**, which are by 1 longer than the data extent.
- Support for scattered data and **non-destructive binning**.
  This includes first and foremost **event data**, a particular form of sparse data with arrays of random-length lists, with very small list entries.
- Support for **masks stored with data**.
- Internals written in C++ for better performance (for certain applications), in combination with Python bindings.

Generic functionality of scipp is provided in the **scipp** Python package.
In addition, more specific functionality is made available in other packages.
Examples for this are `scippneutron <https://scipp.github.io/scippneutron>`_ for handling data from neutron-scattering experiments,
and `ess <https://scipp.github.io/ess>`_ for dealing with the specifics of neutron instruments at ESS.

News
----

- [|SCIPP_RELEASE_MONTH|] scippnexus-|SCIPP_VERSION| `has been released <about/release-notes.rst>`_.

Where can I get help?
---------------------

We strive to keep our documentation complete and up-to-date.
However, we cannot cover all use-cases and questions users may have.

We use GitHub's `discussions <https://github.com/scipp/scipp/discussions>`_ forum for questions
that are not answered by these documentation pages.
This space can be used to both search through problems already met/solved in the community
and open new discussions if none of the existing ones provide a satisfactory answer.

Documentation
=============

.. toctree::
   :caption: Getting Started
   :maxdepth: 3

