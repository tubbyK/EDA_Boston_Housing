# as learned on https://www.kaggle.com/prasadperera/the-boston-housing-dataset

from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_Boston():
    def __init__(self):
        self.data = None
        self.fig_index = 0

    def run(self):
        self.load_data()
        self.box_plot_()
        self.outliers()
        self.dist_plot_()
        self.heat_map_()
        self.plot_vs_target()
        for i in range(1,self.fig_index+1):
            plt.figure(i)
            plt.show()
        import time
        time.sleep(10)

    def load_data(self):
        data = datasets.load_boston()
        self.data = pd.DataFrame(data=np.c_[data.data, data.target],
                                 columns=data.feature_names.tolist() + ['MEDV'])
        print('data loaded')
        print(self.data.shape)
        print(self.data.head())
        print(self.data.describe())

    def box_plot_(self):
        self.fig_index += 1
        plt.figure(self.fig_index)
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for k, v in self.data.items():
            sns.boxplot(y=k, data=self.data, ax=axs[index])
            index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    def outliers(self):
        for k, v in self.data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(self.data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
    
    def dist_plot_(self):
        self.fig_index += 1
        plt.figure(self.fig_index)
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for k, v in self.data.items():
            sns.distplot(v, ax=axs[index])
            index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    def heat_map_(self):
        self.fig_index += 1
        plt.figure(self.fig_index)
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.data.corr().abs(), annot=True)

    def plot_vs_target(self):
        self.fig_index += 1
        plt.figure(self.fig_index)
        min_max_scaler = preprocessing.MinMaxScaler()
        column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
        x = self.data.loc[:, column_sels]
        y = self.data['MEDV']
        x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for i, k in enumerate(column_sels):
            sns.regplot(y=y, x=x[k], ax=axs[i])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


if __name__ == '__main__':
    e = EDA_Boston()
    e.run()