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
        sns.heatmapmap(self.train_data.corr(), annot=True, center=0, cmap='vlag')
        sns.clustermap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
        plt.show()
