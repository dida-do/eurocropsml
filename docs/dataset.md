# EuroCropsML Dataset

## Project Background 
```{include} ../README.md
:start-after: 'project-background-begin -->'
:end-before: '<!-- project-background-end'
```

## Data Acquisition

In order to receive the reflectance data, the following steps were taken separately for each country:
1. {any}`.SAFE-files collection<eurocropsml.acquisition.collector>`: Request the necessary `.SAFE`-files for 2021 via an API request to the [EO-lab Finder](https://finder.eo-lab.org/).
2. {any}`.SAFE-files collection<eurocropsml.acquisition.collector>`: Join parcels and `.SAFE`-files by their geometries.
3. {any}`Band image path collection<eurocropsml.acquisition.copier>`: Move all necessary `.SAFE`-files to a local directory to fasten up polygon clipping. Collect the individual band image paths of each `.SAFE`-file.
4. {any}`Polygon clipping<eurocropsml.acquisition.clipper>`: Clip parcels from the `.SAFE`-files to obtain time series of corresponding reflectance data. As the dataset is intended to be used for crop type classification, we calculate the median pixel value for each Sentinel-2 band, as also done in the [tiny EuroCrops dataset](https://arxiv.org/abs/2106.08151).
5. {any}`NUTS regions<eurocropsml.acquisition.region>`: Add NUTS1-NUTS3 regions. The shapefiles for the NUTS-regions have been obtained from [eurostat](https://ec.europa.eu/eurostat/de/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts).

:::{note}
During step one, some parcels that lie at the intersection of two or multiple raster tiles, are assigned to all of them.
In this case, only the raster tile with the lowest cloud coverage is kept for the following steps.
Thus, during clipping, only parts of the polygon are clipped and included in the calculation of the median pixel value.
However, since we are only relying on the median pixel value and not on individual pixel values, clipping only a part of the polygon in rare edge cases is sufficient.
:::

![Data Acquisition Pipeline.](_static/acquisition-pipeline.jpg){height=300px}

We provide all scripts that are necessary to perform the above steps. 

:::{note}
The scripts could be adapted accordingly in order to get similar data for other countries present in EuroCrops, as long as you have access to the necessary `.SAFE`-files. Please create a {gitref}`separate configuration file<eurocropsml/configs/acquisition/cfg/>` for this.
The {any}`config module<eurocropsml.acquisition.config>` already contains the necessary information for the other available EuroCrops countries. Please refer to the [official EuroCrops reference dataset](https://zenodo.org/records/10118572) for more reference data.
:::

To run the data collection, you can use the provided {doc}`command-line interface (CLI)<cli>`.

The following commands provide further assistance:
```console
$ eurocropsml-cli --help
```

```console
$ eurocropsml-cli acquisition eurocrops --help
```

The {gitref}`default<eurocropsml/configs/acquisition/config.yaml>` configuration collects data for Portugal. 

:::{note}
If you want to get the data for another country, please first create a new {gitref}`configuration file<eurocropsml/configs/acquisition/cfg/>`. You can then simply replace the default configuration with the one you created.

For example:
```console
$ eurocropsml-cli acquisition eurocrops get-data cfg=estonia
```
:::

## Data Preprocessing
The collected data needs further preprocessing in order to be used with most machine learning models.

To run the data preprocessing, you can use the provided {doc}`command-line interface (CLI)<cli>`.

The following command provides further assistance:
```console
$ eurocropsml-cli datasets eurocrops --help
```

During preprocessing, each data point is saved separately as an `.npz`-file along with metadata such as the spatial coordinates of the center of the parcel and the date of each observation.

### Cloud Removal
Additionally, we perform a cloud removal step following the scene classification approach of the [Level-2A Algorithm](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview). To detect clouds, we rely on the brightness thresholds of the red band (B4). If the median reflectance of the band is lower than the T1 threshold (0.07), we consider it as cloud-free and assign a cloud probability of 0%. If it is higher than the T2 threshold (0.25), it is considered cloudy and is assigned a cloud probability of 100%. Similarly, we assign probabilities between 0 and 100% and remove all observations with a cloud probability greater than 50%. The removal of the cloudy observations can be turned off in the preprocess config.

### Further Notes
Please note that when creating training/validation splits for machine learning algorithms, there is an option for downsampling the class `pasture_meadow_grassland_grass` during the pre-training phase of a transfer-learning scenario to the median frequency of all other classes. The downsampling can be turned off in the {gitref}`split-config<eurocropsml/configs/dataset/split>` by removing the `meadow_class` parameter.