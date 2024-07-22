API Reference
=============

.. mermaid::
    :align: center

     %%{init: {'theme':'base'}}%%
    flowchart TB

        package(eurocropsml) --> subpackage1(eurocropsml.acquisition) & subpackage2(eurocropsml.dataset)

The API reference gives an overview over all public modules, classes, and functions within in the `eurocropsml` package.
It is organized as two main sub-packages:
The first focuses on acquiring the raw data.
The second is used for building ready-to-use machine learning datasets from the obtained raw data.

The usage of individual sub-packages and modules within our data processing is illustrated below.

.. image:: _static/acquisition-pipeline.png
   :height: 300px
   :alt: Data Acquisition Pipeline.
   :align: center

We also have a separate :doc:`CLI Reference<cli>` giving an overview of how to interact with the `eurocropsml` package via a command line interface.


.. rubric:: Package Content

.. autosummary::
    :toctree: api
    :template: custom-module.rst
    :recursive:

    eurocropsml.acquisition
    eurocropsml.dataset



