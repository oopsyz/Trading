import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quantstats as qs
import gymnasium as gym
from envs.stocks_RL_env import StocksRLEnv

from gym_anytrading import datasets as ds
from gym_anytrading.envs import Actions
from stable_baselines3 import A2C

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

df = pd.read_csv('data/STOCK_US_XNYS_VZ.csv')
df.set_index('Date', inplace=True)
#df=ds.STOCKS_GOOGL

print(df.head())

window_size = 10
start_index = window_size
end_index = len(df)
print("end_index=",end_index)
#env = gym.make('stocks-v0', df=df, frame_bound=(start_index,end_index), window_size=window_size)
render_mode=None

env = StocksRLEnv( df=df,
    window_size=window_size,
    frame_bound=(start_index, end_index),
    render_mode=render_mode)

state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, truncated, info = env.step(action)
    if done or truncated: 
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()