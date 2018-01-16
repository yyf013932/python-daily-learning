import pandas as pd
import numpy as np
from fbprophet import Prophet as prp

df = pd.read_csv('d:/resources/tc_yancheng/train_20171215.txt', delimiter='\t')

m = prp()
m.fit(df)
