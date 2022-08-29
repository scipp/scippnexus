.. _release-notes:

Release Notes
=============

v0.2.0 (August 2022)
--------------------

Features
~~~~~~~~

* :meth:`scippnexus.NXobject.__getitem__` now accepts classes such as :class:`scippnexus.NXlog` or :class:`scippnexus.NXdata` as key and returns all direct children with an ``NX_class`` attribute matching the provided class `#48 <https://github.com/scipp/scipp/pull/48>`_.
* Added "dynamic" properties to :class:`scippnexus.NXobject`, to access unique children such as entry or instrument `#49 <https://github.com/scipp/scipp/pull/49>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* The ``NX_class`` enum has been removed. Use classes such as ``NXlog`` as keys from now on `#48 <https://github.com/scipp/scipp/pull/48>`_.
* The ``by_nx_class`` method has been removed `#48 <https://github.com/scipp/scipp/pull/48>`_.

Bugfixes
~~~~~~~~

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.1.3 (June 2022)
------------------

Bugfixes
~~~~~~~~

* Fixed exception when slicing with single integer (instead of a range) `#36 <https://github.com/scipp/scipp/pull/36>`_.
* Fixed slicing with bin-edge coords, which previously dropped the upper bound `#36 <https://github.com/scipp/scipp/pull/36>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`

v0.1.0 (May 2022)
-----------------

Features
~~~~~~~~

This is the initial non-experimental release of scippnexus.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

Contributing Organizations
--------------------------
* :sup:`a`\  `European Spallation Source ERIC <https://europeanspallationsource.se/>`_, Sweden
