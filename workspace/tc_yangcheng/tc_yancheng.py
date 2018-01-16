import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('d:/resources/tc_yancheng/train_20171215.txt', delimiter='\t')

# sns.factorplot('brand', 'cnt', data=df)

sns.FacetGrid(df, hue='brand').map(plt.plot, 'cnt')

sns.plot