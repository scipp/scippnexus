# Our Interpretation of the NeXus Format

Last update: January 2022

## Introduction

This document represents our current interpretation of the [NeXus Format](https://www.nexusformat.org/) (NF).
The need for an interpretation is an unfortunate truth.
The reason for this is partially history, as the NF has evolved over time, whereas files from older versions of the NF are still used.
The second reason is unclear and/or incomplete formulations in the NF description that leave things open to interpretation or require reading between the lines.

A majority of this document is based on personal communication with Tobias Richter (TR) from the NeXus committee.
Nevertheless, most of this still represents our own interpretation.

## [NXroot](https://manual.nexusformat.org/classes/base_classes/NXroot.html)

The `NX_class` attribute for the top-level of a NeXus hierarchy (NH, often a file) should be `NXroot`.
However, this has been forgotten in the reference implementation and thus most file-writers do not include it.
It is therefore to be considered implicit.

## Relation between [NXdata](https://manual.nexusformat.org/classes/base_classes/NXdata.html) and [NXevent_data](https://manual.nexusformat.org/classes/base_classes/NXevent_data.html) with "subclasses"

The NF does not formally specify subclassing but, according to TR and vague hints in the manual, several classes are "similar" to [NXdata](https://manual.nexusformat.org/classes/base_classes/NXdata.html).
This probably includes [NXdetector](https://manual.nexusformat.org/classes/base_classes/NXdetector.html), [NXlog](https://manual.nexusformat.org/classes/base_classes/NXlog.html), and [NXmonitor](https://manual.nexusformat.org/classes/base_classes/NXmonitor.html) but there are likely more.

By extension, [NXevent_data](https://manual.nexusformat.org/classes/base_classes/NXevent_data.html) may take the role of `NXdata` for event-mode monitors or detectors.

It is unclear whether the NF requires fields and attributes of the base class (`NXdata` or `NXevent_data`) as part of the subclass (such as `NXdetector`), or whether `NXdata` or `NXevent_data` should be stored as a child.
We have observed both approaches in practice.
Therefore, we go with the following interpretation:

1. *Both* approaches are permitted.
2. For a concrete group in a NH, there must not be more than one `NXdata` or `NXevent_data` child (or field thereof) within the group.
   If there is more than one, the group is considered invalid.
3. Certain classes pre-define aspects that also `NXdata` could deliver.
   For example, `NXdata` uses the [`signal` attribute](https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-signal-attribute) to define the field providing the signal array.
   For `NXdetector` the NF pre-defines that its signal is `data`, so if that is found it will be used, even if no `signal` attribute is present.

More concretely this means that, e.g., for loading an `NXdetector` from a NH, the implementation will:

1. Find all `NXdata` children.
2. Find all `NXevent_data` children.
3. Search the group for fields defined in `NXdata`.
   This includes looking for the `signal` attributes on the group and on fields.
4. Search the group for fields defined in `NXevent_data`.
5. Search the group for fields pre-defined in the class that are equivalents of what is defined in `NXdata`, even if the `NXdata` requirements (such as `signal` attributes) are not met.

If the above yields no more than one item, the group can be loaded.

## NXlog as field replacement

According to TR [NXlog](https://manual.nexusformat.org/classes/base_classes/NXlog.html) can replace any field, if the field is time dependent.
This is typically used within [NXsource](https://manual.nexusformat.org/classes/base_classes/NXsource.html), [NXsample](https://manual.nexusformat.org/classes/base_classes/NXsample.html), [NXdisk_chopper](https://manual.nexusformat.org/classes/base_classes/NXdisk_chopper.html), and others.
Another example is [NXtransformations](https://manual.nexusformat.org/classes/base_classes/NXtransformations.html) for time-dependent positions of beamline components.

## Datetime fields

HDF5 does not support storing date and time information such as `np.datetime64`.
`NXlog` and `NXevent_data` specify specific attributes for fields that have to be interpreted as date and time, in particular [NXlog/time@start](https://manual.nexusformat.org/classes/base_classes/NXlog.html#nxlog-time-start-attribute) and [NXevent_data/event_time_offset@offset](https://manual.nexusformat.org/classes/base_classes/NXevent_data.html#nxevent-data-event-time-offset-field).
No *general* definition or intention is documented in the NF, but according to TR this is nevertheless standard.
Due to the attribute naming mismatch in the two cases where it *is* specified we need to assume that naming is arbitrary.
Therefore, we search *all* attributes of a field for a date and time offset, provided that the field's unit is a time unit.
It is unclear what should be done in the case of multiple matches.
As of April 2022 we ignore the date and time offsets in this case, since guessing which one to use based on the attribute name does not seem desirable.

## Bin edges

For [NXdetector](https://manual.nexusformat.org/classes/base_classes/NXdetector.html) the NF defines a [time_of_flight](https://manual.nexusformat.org/classes/base_classes/NXdetector.html#nxdetector-time-of-flight-field) field, exceeding the data shape by one, i.e., it is meant as bin-edges.
`NXdata` does not appear to allow this explicitly.
Since what is recorded in `NXdetector` may not actually be time-of-flight, in practice this coordinate may be named differently, e.g., `time_offset`.
Therefore, we assume that this is valid in general, i.e., also for other axis values (axis tick labels) that may be defined using the [`axes` attribute](https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-axes-attribute).

According to TR this is also used frequently in some fields/applications of [NXdata](https://manual.nexusformat.org/classes/base_classes/NXdata.html).

## Missing axis labels

[NXdata](https://manual.nexusformat.org/classes/base_classes/NXdata.html) uses the [`axes` attribute](https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-axes-attribute) to define the names of fields that store coordinates for axes.
There is a legacy mechanism where the signal field has an [`axes` attribute](https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-data-axes-attribute) and this should not be used according to the NF.

The `axes` attribute uses `'.'` to define an axis without values, i.e., without a field.
The implication of the above is that there is no way to define the *label* of an axis, unless also values are defined.
For example, an `NXdata` group storing a stack of images may not define a coordinate for the "image" dimension of the signal dataset.
The NF does not specify any other way to define such an axis name.
We therefore have to fall back to a generic and meaningless dimension label.

Note that the `axes` attribute of the signal field could in principle be used to define such axis labels.
The `axes` attribute of the group would then define which axis has a corresponding field with values.
That is, both attributes would work in conjunction, but this is not considered valid under the NF currently.
