import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = ""

x = pd.read_csv("parkinsons_updrs.data")
subj = pd.unique(x['subject#'])
X = pd.DataFrame()

for k in subj:
    # k-th patient data
    xk = x[x['subject#'] == k]
    xk1 = xk.copy()

    xk1.test_time = xk1.test_time.astype(int)
    xk1['g'] = xk1['test_time']
    v = xk1.groupby('g').mean()

    X = pd.concat([X, v], axis=0, ignore_index=True)
features = x.columns
