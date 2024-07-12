# Examples

Below we show some examples of customization options when using `eurocropsml`.
It is recommended to read the introductory {doc}`starting` section of our documentation first.

## Customizing the download directory
By default the `eurocropsml-cli` will download data into a local directory named `data` as specified in the global settings file `eurocropsml/settings.py`.
This configured directory path is **relative** to the `eurocropsml` package installation directory.
This works well, if you obtained `eurocropsml` by cloning the git repository for local development.
But it might not be the desired behavior if you installed `eurocropsml` via `pip` as a dependency in another project.
In this case the relative data paths would become something like `.../lib/python3.10/site-packages/eurocropsml/...`, e.g., within a virtual environment, which is most likely not the place where you would want to store data. 

The data downloading directory can easily be customized, by overwriting the configuration value via setting an environment variable used by the CLI.
The relevant environment variable for the data directory is `EUROCROPS_DATA_DIR` and should be set as an absolute path if you want to make it independent of the package installation directory.

```console
$ export EUROCROPS_DATA_DIR=<absolute-path-to-data-directory>

$ eurocropsml-cli datasets eurocrops download
```

## Customizing additional directories

There are three environment variables that can set custom directory paths for the `eurocropsml-cli`.

 `EUROCROPS_CONFIG_DIR` 
 : Sets the directory where the CLI looks for YAML configurations (default `eurocropsml/configs`).

 `EUROCROPS_DATA_DIR` 
 : Sets the directory where preprocessed and downloaded data is stored (see above) (default `data`).

 `EUROCROPS_ACQUISITION_DIR`
 : Sets the directory where intermediate data is stored during data acquisition and raw data processing (default `acquisition`).

## Obtaining data for different countries

The ready-to-use EuroCropsML dataset provides preprocessed data for Estonia, Latvia, and Portugal.
Using the `eurocropsml` package and `eurocropsml-cli` you can use the same processing-pipeline steps also to obtain analogously preprocessed data for other countries. In order to do so, please make sure to download and unzip the necessary [vector data](https://zenodo.org/records/10118572) to into your `EUROCROPS_DATA_DIR` directory. The [NUTS files](https://ec.europa.eu/eurostat/de/web/gisco/geodata/statistical-units/territorial-units-statistics) from Eurostat can be downloaded manually in advance. If not present, they will be downloaded automatically. The structure of the `meta_data` directory should be as follows:
```console
└── data/
    └── meta_data/
        ├── NUTS/
        │   ├── NUTS_RG_01M_2021_3035
        │   ├── NUTS_RG_01M_2021_3857
        │   └── NUTS_RG_01M_2021_4326
        └── vector_data/
            └── country_folder_from_zenodo
```


The `eurocropsml-cli` will, by default, assume that the Sentinel-2 data is located inside a directory called `/eodata`. However, if the directory where the Sentinel-2 is stored is named differently, the `eodata_dir` argument can be utilized in order to change the parent folder of the `.SAFE`-filepaths returned by the EOLab Finder.

```console
$ eurocropsml-cli acquisition eurocrops get-data +cfg.eodata_dir="personal_eodata_dir"
```
