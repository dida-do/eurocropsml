# Changelog

Version numbers follow the [semantic version](https://semver.org/) (`<major>.<minor>.<patch>`) format.

Changes for the upcoming release can be found (and will be added on merging feature branches) in the next section.

Changes from previous releases are listed below.

## Upcoming Release
- Include hidden coverage report files in CI as Github action artifacts _(see #40)_
- Exit script if automatic acquisition of NUTS files wasn't successful _(see #37)_
- Add acquisition of Sentinel-1 data _(see #35)_
- Add documentation of acquiring and processing Sentinel-1 data _(see #38)_
- Add preprocessing of Sentinel-1 _(see #31)_
- Renaming and moving of Sentinel-2 bands _(see #42)_
- Adjusting split generation _(see #45)_
- Fixing padding mask _(see #50)_
- Fasten up padding to 366 days _(see #54)_
- Radiometric calibration of Sentinel-1 _(see #47)_
- Downloading Sentinel-1 data and make it usable together with Sentinel-2 _(see #43)_

## 0.3.1 (2024-07-29)
- Remove country_code variable in collector downloader _(see #33)_

## 0.3.0 (2024-07-29)
- Ensure that parcel_id column is integer _(see #30)_
- Improve EuroCrops filename definition _(see #27)_
- Ensure that parcel_id_name column is string _(see #25)_
- Remove crs argument from Pyogrio GeoDataFrame when saving to file _(see #24)_
- Fix country code for region module _(see #22)_
- Implement automatic downloading for Eurostat GISCO NUTS files _(see #15)_
- Update Eurostat GISCO NUTS files URL _(see #15)_
- Replace EOLab Finder with EOLab Data Explorer _(see #9)_
- Adjust `shape_dir` for Spain and `shapefile_dir_clean` in general _(see #16)_
- Add `eodata_dir` argument to make Sentinel-2 directory customizable (default is `eodata`) _(see #7)_
- Add splits to split config: 20, 100, 200, 500, "all" _(see #6)_
- Remove `acquisition.analysis module` (redundant) _(see #5)_
- Make documentation building compatible with version `2024.05.06` of `furo` theme for `sphinx` _(see #3 and #4)_
- Remove `sphinx-gitref` dependency for documentation building _(see #3 and #4)_
- Auto generate CLI reference documentation during builds instead of checking it into the repository (_see #3 and #4)_
- Update CI workflow for documentation building to report errors instead of warning _(see #3 and #4)_

## 0.2.0 (2024-06-10)

- Relax the torch version requirement to `torch>=2.0` instead of `torch==2.2.0` _(see #1 and #2)_

## 0.1.0 (2024-04-26)

- Initial release
