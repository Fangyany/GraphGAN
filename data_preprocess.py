import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate


idx = 0
avl = ArgoverseForecastingLoader('./dataset/train_mini/data')
avl.seq_list = sorted(avl.seq_list)  # 显示数据的地址

print(avl[0]) # 读取第一个data
city = copy.deepcopy(avl[idx].city)
"""TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
df = copy.deepcopy(avl[idx].seq_df)  # 传idx，df读出来对于csv的内容


agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))  # 读取时间戳
agt_ti = np.sort(np.unique(df['TRACK_ID'].values))   # 读取轨迹id（每个车有一个id）
print('agt_ts:', agt_ts, len(agt_ts))
print('agt_ti:', agt_ti, len(agt_ti))

mapping = dict()
for i, ts in enumerate(agt_ts):
    mapping[ts] = i
print('mapping', mapping)  # 把时间戳转化成0,1,2,3...

trajs = np.concatenate((
    df.X.to_numpy().reshape(-1, 1),
    df.Y.to_numpy().reshape(-1, 1)), 1)  # 处理轨迹

steps = [mapping[x] for x in df['TIMESTAMP'].values]
steps = np.asarray(steps, np.int64)
objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
keys = list(objs.keys())
obj_type = [x[1] for x in keys]
agt_idx = obj_type.index('AGENT')  # agent作为本车
idcs = objs[keys[agt_idx]]

agt_traj = trajs[idcs]
agt_step = steps[idcs]
del keys[agt_idx]
ctx_trajs, ctx_steps = [], []
for key in keys:
    idcs = objs[key]
    ctx_trajs.append(trajs[idcs])
    ctx_steps.append(steps[idcs])
data = dict()
data['city'] = city
data['trajs'] = [agt_traj] + ctx_trajs
data['steps'] = [agt_step] + ctx_steps

print(data.keys())   # dict_keys(['city', 'trajs', 'steps'])






