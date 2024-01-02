import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

# create environment
env=gym.make('CartPole-v1',render_mode='human')
# reset the environment, 
# returns an initial state

# simulate the environment
episodeNumber=2
timeSteps=1000

#model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=6000)
#model.save("cartpole")
model=A2C.load("cartpole_DQN")
time.sleep(2)
print("Done with training")
for episodeIndex in range(episodeNumber):
    observation,_=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        #random_action=env.action_space.sample()
        random_action, _states=model.predict(observation)
        observation, reward, terminated, interrupted, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.02)
        '''
        if (terminated):
            print("Terminated")
            time.sleep(2)
            break
        '''
env.close()   