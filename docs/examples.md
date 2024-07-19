# Examples

Below we show some examples of customization options when using $\texttt{eurocropsml}$.
It is recommended to read the introductory {doc}`starting` section of our documentation first.

## Customization of directories
There exist several directory variables that are customizable to the needs of the user.

### Customizing the download directory
By default, the `eurocropsml-cli` will download data into a local directory named `data` as specified in the global settings file `eurocropsml/settings.py`.
This configured directory path is **relative** to the $\texttt{eurocropsml}$ package installation directory.
This works well, if you obtained $\texttt{eurocropsml}$ by cloning the git repository for local development.
But it might not be the desired behavior if you installed $\texttt{eurocropsml}$ via `pip` as a dependency in another project.
In this case the relative data paths would become something like `.../lib/python3.10/site-packages/eurocropsml/...`, e.g., within a virtual environment, which is most likely not the place where you would want to store data. 

The data downloading directory can easily be customized, by overwriting the configuration value through the establishment of an environment variable that is used by the CLI.
The relevant environment variable for the data directory is `EUROCROPS_DATA_DIR` and should be set as an absolute path if you want to make it independent of the package installation directory.

```console
$ export EUROCROPS_DATA_DIR=<absolute-path-to-data-directory>

$ eurocropsml-cli datasets eurocrops download
```

### Customizing additional directories

There are three environment variables that can set custom directory paths for the `eurocropsml-cli`.

 `EUROCROPS_CONFIG_DIR` 
 : Sets the directory where the CLI looks for YAML configurations (default `eurocropsml/configs`).

 `EUROCROPS_DATA_DIR` 
 : Sets the directory where preprocessed and downloaded data is stored (see above) (default `data`).

 `EUROCROPS_ACQUISITION_DIR`
 : Sets the directory where intermediate data is stored during data acquisition and raw data processing (default `acquisition`).


### Customizing eo-data directory

The `eurocropsml-cli` will, by default, assume that the Sentinel-2 data is located inside a directory called `/eodata`. However, if the directory where the Sentinel-2 is stored is named differently, the `eodata_dir` argument can be utilized in order to change the parent folder of the `.SAFE`-filepaths returned by the EOLab Data Explorer.

```console
$ eurocropsml-cli acquisition eurocrops get-data +cfg.eodata_dir="personal_eodata_dir"
```

## Customizing the acquisition pipeline
### Obtaining data for different countries

The ready-to-use $\texttt{EuroCropsML}$ dataset provides preprocessed data for Estonia, Latvia, and Portugal.
The $\texttt{eurocropsml}$ package in conjunction with the `eurocropsml-cli` enables the reuse of the processing pipeline steps to obtain analogous preprocessed Sentinel-2 data for other countries. 
In order to do so, it is recommended to create a new configuration file for each country (for example `eurocropsml/configs/acquisition/cfg/eurocrops_country.yaml`) and select it from the command line:
```console
$ eurocropsml-cli acquisition eurocrops get-data cfg=eurocrops_country
``` 

However, individual changes to an existing configuration, such as collecting data only for the month of January for Estonia, can also be achieved via the CLI
```console
$ eurocropsml-cli acquisition eurocrops get-data cfg=estonia cfg.country_config.months="[1,1]"
``` 

:::{note}
Please first make sure to download and unzip the necessary [vector data](https://zenodo.org/records/10118572) into your `EUROCROPS_DATA_DIR` directory. The [NUTS files](https://ec.europa.eu/eurostat/de/web/gisco/geodata/statistical-units/territorial-units-statistics) from Eurostat can be downloaded manually in advance. If not present, they will be downloaded automatically. The structure of the `meta_data` directory should be as follows:
```console
    └── meta_data/
        ├── NUTS/
        │   ├── NUTS_RG_01M_2021_3035
        │   ├── NUTS_RG_01M_2021_3857
        │   └── NUTS_RG_01M_2021_4326
        └── vector_data/
            └── country_folder_from_zenodo
```
:::

### Adjusting multiprocessing parameters
Each country config contains a number of parameters that are utilized for multiprocessing, which may be adjusted in accordance with the available resources. It should be noted that the limitations pertain, in particular, to the {any}`clipping module<eurocropsml.acquisition.clipper>`.
- `workers` (default 16, used in multiple modules): Maximum number of parallel workers used for multiprocessing. This is contingent upon the vailability of central processing units (CPUs). In the event that this exceeds the number of CPUs, the parallel workers will be set to the number of CPUs. However, in instances where the random-access memory (RAM) capacity is insufficient—such as when constructing the argument list prior to clipping—this value can be reduced within the configuration parameters. It is essential to note that this adjustment may result in a slowing of the process. Therefore, it is recommended to only reduce the number of parallel workers when absolutely necessary.

The following two parameters are exclusively used during the clipping process. In the event that the available RAM is insufficient, they can be lowered. It is important to note that this will impede the clipping process, and thus, we again only advise reducing them if absolutely necessary.
- `chunk_size`: Number of chunks that are procssed in parallel.
- `multiplier`: This is used to save intermediate results during the clipping process, thus, preventing the RAM from exceeding its operational limits. Upon the processing of "`multiplier`" data chunks,  have been processed, the current DataFrame is stored.


## Customizing the dataset pipeline
### Customizing the pre-processing pipeline
The pre-processing settings depend on the configuration file that is located within the `eurocropsml.configs.dataset.preprocess` module. Individual changes to an existing parameter value can again be made directly through the CLI. The following command can be used in order to disable the removal of cloudy observations:
```console
$ eurocropsml-cli datasets eurocrops <COMMAND> preprocess.filter_clouds=false
``` 
To adjust the lower and upper thresholds $t_1$ and $t_2$ which determine whether an observation is classified as cloudy or non-cloudy, the user can customize the values.
For example, to set the lower bound to $0.04$ and the upper bound to $0.2$:
```console
$ eurocropsml-cli datasets eurocrops <COMMAND> preprocess.band4_t1=0.04 preprocess.band4_t2=0.2
``` 

If multiple customizations are required, it is advisable to create a new custom configuration `.yaml` file, for instance `eurocropsml/configs/dataset/preprocess/custom_config.yaml` and to select it via the command line:
```console
$ eurocropsml-cli datasets eurocrops <COMMAND> preprocess=custom_config
``` 

### Customizing the dataset utilization
The $\texttt{EuroCropsML}$ dataset allows users to customize options for various crop type classification scenarios, making it suitable for a range of benchmarking applications. Adjustments can be made by creating a custom split configuration within the `eurocropsml.configs.dataset.split` module or by modifying the parameters of existing configurations. Detailed split configuration parameters are listed in the table below.

| __Parameter__ | __Definition__ |
| ------------- | ------------- |
| `base_name` | Base name of the split configuration, used when creating and saving the splits. |
| `data_dir` | Folder inside the data directory where pre-processed data is stored. | 
| `random_seed` | Random seed used for generating training-testing-splits and further random numbers. |
| `num_samples` | Number of samples per class used for the fine-tuning subsets. The default will create the shots currently present on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10629609) for the training set. It will samples 1000 samples for validation and keep all available data from the test set. |
| `meadow_class` | Class that represents the ${\texttt{pasture_meadow_grassland_grass}}$ class. If provided, then this class will be downsampled to the median frequency of all other classes for the pre-training dataset since it represents an imbalanced majority class. |
| `pretrain_classes` | Classes that make up the pre-train dataset. |
| `finetune_classes` | Classes that make up the pre-train dataset. |
| `pretrain_regions` | Regions that make up the pre-train dataset. |
| `finetune_regions` | Regions that make up the fine-tune dataset. |


The class configuration parameters `pretrain_classes` and `finetune_classes` as well as the region parameters `pretrain_regions`and `finetune_regions` can be specified by providing collections of key-value pairs (dictionaries), respectively. 
The keys can be `region`, `regionclass`, or `class`. Each key refers to a different transfer learning use case, as stated in the table below. 
The valid values for the class configurations are the HCAT class labels as used by the $\texttt{EuroCrops}$ reference data, while for the region configurations they are the NUTS region IDs for levels 0--3.

| __Parameter__ | __Use Case__ |
| ------------- | ------------- |
| `region` | The data set is divided into pre-training and fine-tuning subsets based on NUTS regions, with three scenarios:<br>- __Complete Class Overlap:__ All classes between regions overlap, allowing fine-tuning of pre-trained models.<br>- __Partial Class Overlap:__ Some classes overlap between regions, so overlapping classes appear in both pre-training and fine-tuning sets (${\textit{cf.}\,}$ [$\texttt{EuroCropsML}$](https://zenodo.org/doi/10.5281/zenodo.10629609)).<br>- __No Class Overlap:__ No classes overlap between regions, requiring the classifier to adapt to new classes during fine-tuning. |
| `regionclass` | The dataset is divided into pre-train and fine-tune subsets by NUTS regions and subsequently by classes. This setting can be used when classes overlap (partially) between regions, but the pre-train and fine-tune sets should contain different regions and classes. | 
| `class` | The pre-training and fine-tuning subsets are based solely on classes, focusing on knowledge transfer between different sets of classes, not regions.<br>There are two scenarios:<br>- __Partial Overlap:__ Overlapping classes appear in both pre-training and fine-tuning datasets.<br>- __No Overlap:__ The classifier must adapt to entirely new classes during fine-tuning |