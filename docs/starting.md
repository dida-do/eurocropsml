# Getting Started

$\texttt{eurocropsml}$ is a Python package [hosted on PyPI](https://pypi.org/project/eurocropsml/).

## Installation
The recommended installation method is [pip](https://pip.pypa.io/en/stable/)-installing into a virtual environment:

```console
$ python -Im pip install eurocropsml
```

## Usage Guide

The quickest way to interact with the $\texttt{eurocropsml}$ package and get started is using the $\texttt{EuroCropsML}$ dataset via the provided {doc}`command-line interface (CLI)<cli>`.

:::{admonition} Example
:class: note

To **get help** on available commands and options, use
```console
$ eurocropsml-cli --help
```
:::

:::{admonition} Example
:class: note

To **show** the currently used (default) **configuration** for the $\texttt{eurocropsml}$ dataset CLI, use
```console
$ eurocropsml-cli datasets eurocrops config
```
:::

:::{admonition} Example
:class: note

To **download** the $\texttt{EuroCropsML}$ dataset as currently configured, use
```console
$ eurocropsml-cli datasets eurocrops download
```
:::

Alternatively, the dataset can also be manually downloaded from our [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10629609).

---

:::{seealso}
A comprehensive documentation of the CLI can be found in our {doc}`CLI Reference<cli>`.
:::

:::{seealso}
More hands-on examples on using this package and its customization options can be found in our {doc}`Examples<examples>`.
:::

:::{seealso}
For a complete example use-case demonstrating the ready-to-use $\texttt{EuroCropsML}$ dataset in action, please refer to the project's associated [official repository](https://github.com/dida-do/eurocrops-meta-learning) for benchmarking meta-learning algorithms.
:::