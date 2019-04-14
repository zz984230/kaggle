import pandas as pd
import numpy as np


def diff_result():
    bench = pd.read_csv("../data/ori/sampleEntry.csv")
    result = pd.read_csv("../data/result/result.csv")

    bench.loc[bench["Probability"] >= 0.5, "Probability"] = 1
    bench.loc[bench["Probability"] < 0.5, "Probability"] = 0

    result.loc[result["Probability"] >= 0.5, "Probability"] = 1
    result.loc[result["Probability"] < 0.5, "Probability"] = 0

    z = pd.concat((bench["Probability"], result["Probability"]), axis=1)
    z.columns = ["bench", "result"]
    z["diff"] = (z["bench"] == z["result"])
    print(z)

diff_result()
