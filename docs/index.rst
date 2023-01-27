ScippNexus
==========

.. raw:: html

   <span style="font-size:1.2em;font-style:italic;color:#5a5a5a">
      An h5py-like utility for NeXus files with seamless Scipp integration
      </br></br>
   </span>

ScippNexus provides a link between the HDF5-based `NeXus Data Format <https://www.nexusformat.org/>`_ and `scipp <https://scipp.github.io/>`_.
This is possible since NeXus classes (a specification of an HDF5 group and the contained dataset) such as `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html>`_ partially resemble Scipp's `DataArray <https://scipp.github.io/user-guide/data-structures.html#DataArray>`_.

`h5py <https://docs.h5py.org/en/stable/>`_ is a convenient and powerful tool for accessing groups, datasets, and attributes in an HDF5 file.
However, it operates on a lower level than the NeXus class definitions, which typically treat an entire group as a distinct entity.

ScippNexus can obviate the need for lengthy low-level code.
It provides an h5py-like API, but applies the paradigm at the NeXus-class level.
This is especially powerful since a number of concepts of Scipp map well to concepts of NeXus:

- NeXus classes such as NXdata define *dimension scales*, which correspond to *coordinates* in the Scipp terminology.
  From a user's point of view we can thus treat an NXdata group in a file as an entity that can be loaded directly into a :class:`scipp.DataArray`.
- **Labeled dimension** can be used to selectively load slices of an entire NeXus class.
  This is modelled after h5py's support for loading slices of datasets but provides the same convenience and safety as `scipp's slicing <https://scipp.github.io/user-guide/slicing.html>`_ by requiring specification of the dimension to slice by its name, rather than plain axis order.
- **Physical units** are stored with most datasets in a NeXus class and are loaded as unit of the :class:`scipp.DataArray` values or coordinates.

News
----

- [January 2023] scippnexus-23.01.0 has been released.
  This brings support for loading arbitrary groups as :class:`scipp.DataGroup`.
  Furthermore, errors are handled in a more transparent way:
  ScippNexus will now fall back to returning group contents as a :class:`scipp.DataGroup` instead of silently skipping loading certain fields.
- [November 2022] scippnexus-0.4.0 has been released.
  This provides and experimental mechanism for customizing reading and writing of data based on NeXus application definitions.

Get in touch
------------

- If you have questions that are not answered by these documentation pages, ask on `GitHub discussions <https://github.com/scipp/scippnexus/discussions>`_.
  Please include a self-contained reproducible example if possible.
- Report bugs (including unclear, missing, or wrong documentation!), suggest features or view the source code `on GitHub <https://github.com/scipp/scippnexus>`_.

.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 3

   getting-started/installation
   getting-started/quick-start-guide

.. toctree::
   :caption: User Guide
   :hidden:
   :maxdepth: 3

   user-guide/nexus-classes
   user-guide/application-definitions
   user-guide/classes
   user-guide/functions

.. toctree::
   :caption: About
   :hidden:
   :maxdepth: 3

   about/about
   about/release-notes
