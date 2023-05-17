- Install [python](https://www.python.org) version 3.9 or above
- Install the dependencies: `python3 -m pip install scikit-learn pandas matplotlib`
- Download the dataset from [here](https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0257247/GLODAPv2.2022_Atlantic_Ocean.csv)
- Preprocess the dataset using the following command:
```
awk -F',' '!/-9999/' "unformatted_file_path" > "output_file_path"
```

