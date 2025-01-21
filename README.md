# EuroCropsML
*Ready-to-use benchmark dataset for few-shot crop type classification using Sentinel-1 and Sentinel-2 imagery.*

*Part of the [PreTrainAppEO](https://www.asg.ed.tum.de/en/lmf/pretrainappeo/) ("Pre-Training Applicability in Earth Observation") research project.*

<!-- badges begin -->
[![Read the Docs](https://img.shields.io/readthedocs/eurocropsml/latest?style=flat&logo=readthedocs&logoColor=white)](https://eurocropsml.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/eurocropsml?style=flat&logo=pypi&logoColor=white)
](https://pypi.org/p/eurocropsml)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdida-do%2Feurocropsml%2Fmain%2Fpyproject.toml&style=flat&logo=python&logoColor=white)](https://www.python.org)
[![GitHub License](https://img.shields.io/github/license/dida-do/eurocropsml?style=flat)](LICENSE)
[![Zenodo DOI](https://img.shields.io/badge/doi-10.5281/zenodo.10629609-blue?style=flat&logo=doi&logoColor=white)
](https://zenodo.org/doi/10.5281/zenodo.10629609)
<!-- badges end -->

<!-- teaser-begin -->

`EuroCropsML` is a pre-processed and ready-to-use machine learning dataset for crop type classification of agricultural parcels in Europe.
It consists of a total of **706,683** Sentinel-2 and **176,055**  Sentinel-1 multi-class labeled data points with a total of **176** distinct classes.
Each data point contains an annual time series of per parcel median pixel values of Sentinel-1 data and/or Sentinel-2 L1C (top-of-atmosphere) reflectance data for the year 2021. For Sentinel-1, we utilize the C-band Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) data. Imagery is selected based on the orbit type available for the location, either ascending or descending. In terms of polarization, we use Interferometric Wide (IW) mode with VV (vertical polarization emission and reception) and VH (vertical polarization emission and horizontal reception) bands.
The dataset is based on [Version 9](https://zenodo.org/records/10118572) of [`EuroCrops`](https://github.com/maja601/EuroCrops), an open-source collection of remote sensing reference data.

For `EuroCropsML`, we acquired and aggregated data for the following countries:

| Country      | Number of distinct classes | Total number of datapoints for Sentinel-2 | Total number of datapoints for Sentinel-1 | 
|--------------|----------------------------|-------------------------------------------|-------------------------------------------|
| Estonia      | 127                        | 175,906                                   | 176,055                                   | 
| Latvia       | 103                        | 431,143                                   | -                                         | 
| Portugal     | 79                         | 99,634                                    | -                                         | 

![Spatial distribution of labels within Estland and Latvia.](docs/_static/labels_spatial_distribution_EE_LV_nuts3_340.png)
![Spatial distribution of labels within Portugal.](docs/_static/labels_spatial_distribution_PT_nuts3_340.png)

The distribution of class labels differs substantially between the regions of Estonia, Latvia, and Portugal.
This makes  transferring knowledge gained in one region to another region quite challenging, especially if only few labeled data points are available.
Therefore, this dataset is particularly suited to explore transfer-learning methods for few-shot crop type classification. 

The data acquisition, aggregation, and pre-processing steps are schematically illustrated below. A more detailed description is given in the [dataset section](https://eurocropsml.readthedocs.io/en/latest/dataset.html) of our documentation.

![Data Acquisition Pipeline.](docs/_static/acquisition-pipeline-s1s2.png)
<!-- teaser-end -->

## Getting Started

`eurocropsml` is a Python package [hosted on PyPI](https://pypi.org/project/eurocropsml/).

### Installation
The recommended installation method is [pip](https://pip.pypa.io/en/stable/)-installing into a virtual environment:

```console
$ python -Im pip install eurocropsml
```

#### Installation of esa_snappy
During Sentinel-1 pre-processing, we make use of the Python plugin `esa-snappy` which enables us to use the SNAP Java API from Python.
In order to be able to import `esa-snappy` into Python, please follow the following steps:
##### 1. Download your matching `Sentinel Toolboxes` [installation file](https://step.esa.int/main/download/snap-download/).
##### 2. If you are using a VM via ssh connection, ssh into your VM using the `-X`-option to later be able to start the SNAP GUI. E.g.
```console
$ ssh -i ~/.ssh/key -X user@ip-address>
```
For MacOS users:
TODO
##### 3. Start the installation of your downloaded installation file, e.g. for Linux and SNAP version 11.0.0:
```console
$ bash esa-nap_sentinel_linux-11.0.0.sh
```
This should start the GUI. Follow the installation steps. Install all components except the "optical Toolbox" which isn't necessary.  
##### 4. Install the `esa-snappy` plugin (cf. [official installation guide](https://senbox.atlassian.net/wiki/spaces/SNAP/pages/2499051521/Configure+Python+to+use+the+new+SNAP-Python+esa_snappy+interface+SNAP+version+10))
1. Run SNAP Desktop by running `$ ./snap` inside the bin folder of your esa-snap directory, e.g. `../esa-snap/bin/`
2. Open the Plugin Manager in SNAP Desktop (Tools → Plugins in the main menu bar)
3. Select tab `Available Plugins`. Among others, the plugin `ESA SNAPPY` appears in the list
4. Select `ESA SNAPPY`, click `Install`, and follow the installation steps as described in the dialogs.
5. After restart of SNAP Desktop, ‘ESA SNAPPY’ will be visible in the list of installed plugins.

##### 5. Configure the `esa-snappy` plugin (cf. [official installation guide](https://senbox.atlassian.net/wiki/spaces/SNAP/pages/2499051521/Configure+Python+to+use+the+new+SNAP-Python+esa_snappy+interface+SNAP+version+10))
With the `esa-snappy` plugin being installed, open a command line window at the bin folder of the SNAP installation directory. We recommend to directly place the plugin in your environment's site packages. In order to do so, type
Unix/MacOS:
```console
$ ./snappy-conf <python-exe> <esa_snappy-dir>
```
Windows:
```console
$ snappy-conf <python-exe> <esa_snappy-dir>
```
with `<esa_snappy-dir>` being for example the `..\lib\python3.10\site-packages` folder of your Python environment.
##### 6. `esa-snappy` can now be imported via
```python
import esa-snappy
```



### Usage Guide
The quickest way to interact with the `eurocropsml` package and get started is to use the `EuroCropsML` dataset is via the provided command-line interface (CLI).

For example, to **get help** on available commands and options, use
```console
$ eurocropsml-cli --help
```

To **show** the currently used (default) **configuration** for the `eurocropsml` dataset CLI, use
```console
$ eurocropsml-cli datasets eurocrops config
```

To **download** the EuroCropsML dataset as currently configured, use
```console
$ eurocropsml-cli datasets eurocrops download
```

Alternatively, the dataset can also be manually downloaded from our [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10629609).

A comprehensive documentation of the CLI can be found in the [CLI Reference](https://eurocropsml.readthedocs.io/en/latest/cli.html) section of our documentation.

For a complete example use-case demonstrating the ready-to-use EuroCropsML dataset in action, please refer to the project's associated [official repository](https://github.com/dida-do/eurocrops-meta-learning) for benchmarking meta-learning algorithms.

## Project Information

The `eurocropsml` code repository is released under the [MIT License](LICENSE).
Its documentation lives at [Read the Docs](https://eurocropsml.readthedocs.io/en/latest/), the code on [GitHub](https://github.com/dida-do/eurocropsml) and the latest release can by found on [PyPI](https://pypi.org/project/eurocropsml/).
It is tested on Python 3.10+.

If you would like to contribute to `eurocropsml` you are most welcome. We have written a [short guide](CONTRIBUTING.md) to help you get started.

### Background

<!-- project-background-begin -->
The EuroCropsML dataset and associated `eurocropsml` code repository are provided and developed as part of the joint [PretrainAppEO](https://www.asg.ed.tum.de/en/lmf/pretrainappeo/) research project by the chair of [Remote Sensing Technology](https://www.asg.ed.tum.de/en/lmf/home/) at Technical University Munich and [dida](https://dida.do/).
<!-- project-background-middle -->

The goal of the project is to investigate methods that rely on the approach of pre-training and fine-tuning machine learning models in order to improve generalizability for various standard applications in Earth observation and remote sensing.

The ready-to-use EuroCopsML dataset is developed for the purpose of improving and benchmarking few-shot crop type classification methods.

`EuroCropsML` is based on [Version 9](https://zenodo.org/records/10118572) of [`EuroCrops`](https://github.com/maja601/EuroCrops), an open-source collection of remote sensing reference data for agriculture from countries of the European Union.
<!-- project-background-end -->

<!-- further-info-begin -->
## Citation
If you use the `EuroCropsML` dataset or `eurocropsml` code repository in your research, please cite our project as follows:

**Plain text**
```text
Reuss, J., & Macdonald, J. (2024). EuroCropsML [dataset]. Zenodo. https://doi.org/10.5281/zenodo.10629610
```
**Bibtex**
```text
@misc{reuss_macdonald_eurocropsml_2024,
  author       = {Reuss, Joana and Macdonald, Jan},
  title        = {EuroCropsML},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10629610},
  url          = {https://doi.org/10.5281/zenodo.10629610}
}
```

## Acknowledgments & Funding
The [PreTrainAppEO](https://www.asg.ed.tum.de/en/lmf/pretrainappeo/) research project is funded by the German Space Agency at DLR on behalf of the Federal Ministry for Economic Affairs and Climate Action (BMWK).
<!-- further-info-end -->