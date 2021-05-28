============
Introduction
============

``spudtr`` provides some special purpose Python 3 functions for
transforming ``pandas.DataFrame`` objects organized in a particular
way. It leans heavily on ``numpy`` and ``scipy``.

The emphasis is operations on multichannel synchronous discrete time
series instrument recordings, such as multi-sensor arrays.

The origin is experimental EEG recordings where the interest is in the
brain activity before, during, and after stimulus and response events
which are logged and time-stamped on recording channels concurrently
with the EEG. Additional data that plays a role in analysis and
modeling such as quality control codes and other experimental
variables is merged in as additional columns alongside the other data
streams which makes a tidy structure for feeding in the analysis
pipelines of primary interest.

This leads naturally to snipping apart longish data logger recordings
into a 2-D tabular data structure that contains a vertical stack of
fixed-length intervals or "epochs" of recordings, indexed by epoch
and time stamped (rows) :math:`\times` parallel
data streams (columns).

This is the spudtr epochs data format (``pandas.DataFrame``) that most
of the functions ingest and transform. Some helper functions like FIR
filtering are agnostic about epochs and just operate on time series
data.

The :ref:`user_guide` has worked examples for some common use cases.

The :ref:`reference` lists all the available functions and documentation,
with links to the source with more helper functions under the hood.
