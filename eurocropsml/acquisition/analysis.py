"""Analyse the dataset and collect class count statistics."""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


def get_country_class_counts(
    output_dir: Path,
    vector_dir: Path,
) -> None:
    """Get counts per class for each country.

    Args:
        output_dir: Directory to save the results.
        vector_dir: Directory where EuroCrops vector data is stored.

    Raises:
        ValueError: If specified country is not available.
    """

    final_df: pd.DataFrame = pd.DataFrame(columns=["EC_hcat_c"])
    for folder in vector_dir.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.suffix == ".shp":
                    shapefile: gpd.GeoDataFrame = pyogrio.read_dataframe(file)
                    counts = shapefile["EC_hcat_c"].value_counts().reset_index()
                    counts.columns = ["EC_hcat_c", folder.name]
                    final_df = pd.merge(final_df, counts, on="EC_hcat_c", how="outer")
                    break

    cols = final_df.columns
    final_df[1:] = final_df[1:].fillna(0)
    final_df[cols[1:]] = final_df[cols[1:]].astype(int)
    final_df.to_pickle(output_dir.joinpath("2021_class_counts.pkg"))
    final_df.to_csv(output_dir.joinpath("2021_class_counts.csv"))


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, help="Directory to save the results.")

    conf = parser.parse_args()
    config = vars(conf)
    for k, v in config.items():
        if "mlp" in k:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config[k] = list(map(int, v.split(",")))

    get_country_class_counts(
        output_dir=Path(config["output_dir"]),
        vector_dir=current_dir.parents[0].joinpath("vectordata"),
    )
