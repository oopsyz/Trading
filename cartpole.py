import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# create environment
env=gym.make('CartPole-v1',render_mode='human')
#env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

# reset the environment, 
# returns an initial state

# simulate the environment
episodeNumber=2
timeSteps=1000

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=6000)
model.save("cartpole")
#model=A2C.load("cartpole")
time.sleep(2)
print("Done with training")

for episodeIndex in range(episodeNumber):
    observation,_=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        #random_action=env.action_space.sample()
        random_action, _states=model.predict(observation)
        print("timeindex:{}  action:{}".format(timeIndex,random_action))
        observation, reward, terminated, interrupted, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.02)
        if (terminated):
            print("Terminated")
            time.sleep(2)
            break
        
env.close()   