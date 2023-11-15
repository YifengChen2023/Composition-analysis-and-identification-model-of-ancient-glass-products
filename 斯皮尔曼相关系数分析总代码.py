# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt  # 可视化
import seaborn as sns  # 可视化
import scipy

df = pd.read_csv('T4-PbBa.csv')  # 读取数据

_, ax = plt.subplots(figsize=(12, 10))

corr = df.corr(method='spearman')
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(
    corr,
    cmap=cmap,
    square=True,
    cbar_kws={'shrink': .9},
    ax=ax,
    annot=True,
    annot_kws={'fontsize': 9})
plt.savefig('PbBa-spearman.png')
plt.show()
