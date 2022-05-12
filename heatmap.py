import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Heatmap:
    def __init__(self, train_data):
        self.train_data = train_data

    def heatmap(self):
        matrix = self.train_data.corr().round(2)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
        sns.clustermap(self.train_data, annot=True, center=0, cmap='vlag')
        plt.show()
