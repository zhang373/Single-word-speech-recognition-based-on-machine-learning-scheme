import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file):
    sp = pd.read_csv(file,header=None)                                          # sp是一个DataFrame
    #print(sp.sample(frac=1))
    databas= sp.iloc[:,1:173]
    label = sp.iloc[:,0]
    #print(pd.DataFrame(databas).shape,pd.DataFrame(databas).sample(frac=1),pd.DataFrame(label).shape,pd.DataFrame(label).sample(frac=1))
    databas = databas.values.astype('float32')
    label = label.values.astype('int64')
    print(type(databas),type(label))

    scale = StandardScaler()
    #fit()求训练集的均值、方差、最大值、最小值等训练集固有的属性。
    #transform()在fit的基础上，进行标准化，降维，归一化等操作
    #fit_transform是fit和transform的组合
    databas = scale.fit_transform(databas)
    databas = torch.from_numpy(databas)
    label = torch.from_numpy(label)
    return databas, label
