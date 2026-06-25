"""Prepare DiffPuter's 'california' dataset (NOT auto-downloaded by their
download_and_process.py -- california is not a UCI set).

Their Info/california.json (num_col_idx 0-8, target_col_idx 9, cat []) matches the StatLib
'California Housing Prices' layout: 9 numerical columns (longitude .. median_house_value)
plus ocean_proximity as the target -- NOT sklearn's 8-feature fetch_california_housing.
So we fetch that CSV, dropna (total_bedrooms has NaNs), make the 70/30 split with their
seed (1234), and generate the masks with their own generate_mask.

  python prep_california.py /path/to/DiffPuter     # run inside the diffputer env

This is the candidate format; the reproduction gate confirms it (does DiffPuter reproduce
its published california RMSE 0.571? if not, adjust the source/columns here).
"""
import os, sys, numpy as np, pandas as pd
from urllib import request

DP = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DS = os.path.join(DP, "datasets", "california")
os.makedirs(DS, exist_ok=True)
URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/master/"
       "datasets/housing/housing.csv")

if not os.path.exists(f"{DS}/data.csv"):
    tmp = "/tmp/ca_housing.csv"
    print(f"downloading California Housing from {URL}")
    request.urlretrieve(URL, tmp)
    df = pd.read_csv(tmp).dropna().reset_index(drop=True)        # total_bedrooms has NaNs
    # columns are already [9 numerical | ocean_proximity], matching california.json
    df.to_csv(f"{DS}/data.csv", index=False)
    n = len(df); idx = np.arange(n)
    np.random.seed(1234); np.random.shuffle(idx)                # same seed as their split
    ntr = int(n * 0.7)
    df.iloc[idx[:ntr]].to_csv(f"{DS}/train.csv", index=False)
    df.iloc[idx[-(n - ntr):]].to_csv(f"{DS}/test.csv", index=False)
    print(f"california data: {n} rows (train {ntr}, test {n - ntr}); cols={list(df.columns)}")
else:
    print("california data.csv already present")

# masks (MCAR / MAR / MNAR x 10) via DiffPuter's own generate_mask
if not os.path.isdir(f"{DS}/masks"):
    sys.path.insert(0, DP); cwd = os.getcwd(); os.chdir(DP)
    from generate_mask import generate_mask
    for mt in ["MCAR", "MAR", "MNAR_logistic_T2"]:
        generate_mask(dataname="california", mask_type=mt, mask_num=10, p=0.3)
    os.chdir(cwd)
    print("california masks generated (MCAR / MAR / MNAR x10)")
print("california ready.")
