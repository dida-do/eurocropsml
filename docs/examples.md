# Examples

Below we show some examples of customization options when using `eurocropsml`.
It is recommended to read the introductory {doc}`starting` section of our documentation first.

## Customizing the download directory

By default the `eurocropsml-cli` will download data into a local directory named `data` as specified in the {gitref}`eurocropsml/settings.py`.
This configured directory path is **relative** to the `eurocropsml` package installation directory.
This works well, if you obtained `eurocropsml` by cloning the git repository for local development.
But it might not be the desired behavior if you installed `eurocropsml` via `pip` as a dependency in another project.
In this case the relative data paths would become something like `.../lib/python3.10/site-packages/eurocropsml/...`, e.g., within a virtual environment, which is most likely not the place where you would want to store data. 

The data downloading directory can easily be customized, by overwriting the configuration value via setting an environment variable used by the CLI.
The relevant environment variable for the data directory is `EUROCROPS_DATA_DIR` and should be set as an absolute path if you want to make it independent of the package installation directory.

```console
$ export EUROCROPS_DATA_DIR=<absolute-path-to-data-directory>

$ eurocropsml-cli datasets eurocrops download
$ eurocropsml-cli datasets eurocrops build-splits split=default
$ eurocropsml-cli datasets eurocrops build-splits split=portugal
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
Using the `eurocropsml` package and `eurocropsml-cli` you can use the same processing-pipeline steps also to obtain analogously preprocessed data for other countries (provided there is access to SHAPE files etc.)