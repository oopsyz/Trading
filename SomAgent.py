import numpy as np
import torch
from VZsampleEnv import MobilePhoneCarrierEnv, Actions
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import optuna

# MultiInputPolicy, MlpLstmPolicy
config={
  "learning_rate":0.005,
  "ent_coef":0.9,
  "policy":'MlpLstmPolicy',
}


env = MobilePhoneCarrierEnv()
if config["policy"] == "MlpLstmPolicy":
  env = FlattenObservation(env)  #used with MlpLstmPolicy

#env = Monitor(env)  # Wrap the environment with Monitor
#env = DummyVecEnv([lambda: env])  # Wrap the environment in a DummyVecEnv

# Check for GPU availability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int("n_epochs", 100, 500)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # ... other hyperparameters
    model = A2C("MultiInputPolicy", env, verbose=0, learning_rate=lr, n_epochs=n_epochs, batch_size=batch_size)
    model.learn(total_timesteps=10000)  # Adjust total_timesteps as needed
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return -mean_reward  # Minimize negative mean reward

def _Random_Agent():
  t = 0
  observation = env.reset()
  while True:
    t += 1
    env.render()
    #print(observation)
    action = env.action_space.sample()
    print("Doing ", Actions.get_action_type(action))
    observation, reward, done, terminated, info = env.step(action)
    if done:
      env.render()
      print(f"***Episode finished after {t+1} timesteps***")
      observation = env.reset()
      break
  input("hit enter")


env.reset()
#env.step(4)

#_Random_Agent()

#train model
train=False  #set to False to just load existing without training
if(train):
  #Hyperparameter tuning
  '''
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=20)  # Adjust n_trials as needed
  best_params = study.best_params
  model = A2C('MultiInputPolicy', env, verbose=1, **best_params, tensorboard_log="./tensorboard_logs/")
  '''
  # static hyperparameters
  if config["policy"] == "MlpLstmPolicy":
    model = RecurrentPPO(config["policy"], env, verbose=1, device=device, learning_rate=config["learning_rate"], batch_size=128, ent_coef=config["ent_coef"], tensorboard_log="./tensorboard_logs/")
  else:
    model = PPO(config["policy"], env, verbose=1, device=device, learning_rate=config["learning_rate"], batch_size=128, ent_coef=config["ent_coef"], tensorboard_log="./tensorboard_logs/")

  model.learn(total_timesteps=500000)
  model.save("activation")
  print("Model Saved")
  del model

finetune=False
entropy_coef=0.1
while finetune:
  params = { 'learning_rate': 0.001, 'n_steps': 1024, 'ent_coef': entropy_coef, 'batch_size': 128, 'n_epochs': 5 }
  if config["policy"] == "MlpLstmPolicy":
    model=RecurrentPPO.load("activation", env, custom_objects=params, device=device)
  else:
    model=PPO.load("activation", env, custom_objects=params, device=device)

  print(f"Hyper Params: lr:{model.learning_rate}; batch size:{model.batch_size}; ent_coef:{model.ent_coef}")
  model.learn(total_timesteps=20000)
  model.save("activation")
  print("Retrained model saved")
  print(f"Current ent_coef: {model.ent_coef}")
  del model
  user_input=input("What is the new value (type 'stop' to stop)?")
  if user_input == 'stop':
     break
  entropy_coef = float(user_input.strip())
  print(f"Using ent_coef: {entropy_coef}")

#load model for prediction   
if config["policy"] == "MlpLstmPolicy":
  model=RecurrentPPO.load("activation", device=device)
else:
  model=PPO.load("activation", device=device)

obs,_ = env.reset()
#print("Reset obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["new_mdn_status"])


steps=0
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

for i in range(300):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
    #action = env.action_space.sample()
    print("Doing ", Actions.get_action_type(action))
    obs, reward, done, terminated, info = env.step(action)
    episode_starts = done
    steps +=1
    #print("episod ", i, "obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["mdn_status"],"reward:", reward, done)
    #print("mdn related to:", obs["mdn2sim"]," use existing? ",obs["use_existing_mdn"])

    if isinstance(env, gym.Wrapper):
      env.env.render(mode="human")
    else:
      env.render(mode="human")

    if done:
      print(f"Completed in {steps} steps")
      obs,_ = env.reset()
      steps=0
      lstm_states = None
      num_envs = 1
      # Episode start signals are used to reset the lstm states
      episode_starts = np.ones((num_envs,), dtype=bool)
      #break

#env.step(env.action_space.sample())
