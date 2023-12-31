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

df = pd.read_csv('data/STOCK_US_XNYS_VZ.csv',thousands=",", decimal=".")
df['Date']=pd.to_datetime(df["Date"])
df.sort_values('Date',ascending=True, inplace=True)
df.set_index('Date', inplace=True)


#df=ds.STOCKS_GOOGL
print(df.head())

window_size = 10
start_index = window_size
end_index = len(df)
#env = gym.make('stocks-v0', df=df, frame_bound=(start_index,end_index), window_size=window_size)
render_mode=None

env = StocksRLEnv( df=df,
    window_size=window_size,
    frame_bound=(start_index, end_index),
    render_mode=render_mode)

action_stats = {Actions.Sell: 0, Actions.Buy: 0}

#use random sample to test env
state = env.reset()
while True: 
    action = env.action_space.sample()
    action_stats[Actions(action)] += 1
    n_state, reward, done, truncated, info = env.step(action)
    if done or truncated: 
        print("Action status:", action_stats)
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

#train model
state = env.reset(seed=2024)
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=50000)

action_stats = {Actions.Sell: 0, Actions.Buy: 0}

observation, info = env.reset(seed=2023)

while True:
    # action = env.action_space.sample()
    action, _states = model.predict(observation)
    action_stats[Actions(action)] += 1
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    if done:
        break

env.close()

print("action_stats(trained):", action_stats)
print("info:", info)

plt.figure(figsize=(16, 6))
env.unwrapped.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.unwrapped.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='SB3_a2c_quantstats.html')