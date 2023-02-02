import os
import pandas as pd
def merge_data():
    socratic_fnames = sorted([fname for fname in os.listdir() if "socratiq_" in fname])
    data = []
    for socratic_fname in socratic_fnames:
        data.append(pd.read_json(socratic_fname))
    return pd.concat(data)
  
if __name__ == "__main__":
  df = merge_data()
  df.to_json("./socratiq.json")
