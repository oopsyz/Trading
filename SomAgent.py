import torch
from VZsampleEnv import MobilePhoneCarrierEnv, Actions
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import obs_as_tensor
import optuna

# Create an environment with a discrete action space of 1
env = MobilePhoneCarrierEnv()

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int("n_epochs", 100, 500)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # ... other hyperparameters
    model = A2C("MultiInputPolicy", env, verbose=0, learning_rate=lr, n_epochs=n_epochs, batch_size=batch_size)
    model.learn(total_timesteps=10000)  # Adjust total_timesteps as needed
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return -mean_reward  # Minimize negative mean reward

def _test_case1():
    print("Using ********** test case****")
    new_state = env.observation_space.sample()
    new_state["address_validation_status"] = 1
    new_state["device_validation_status"] = 1
    new_state["sim_validation_status"] = 1
    #
    new_state["new_mdn_status"] = 3
    new_state["existing_mdn_status"] = 0
    new_state["payment_status"] = 1
    new_state["use_existing_mdn"] = 1
    return new_state

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


env.reset()
#env.step(4)

#_Random_Agent()

#train model
train=True  #set to False to just load existing without training
if(train):
  #Hyperparameter tuning
  '''
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=20)  # Adjust n_trials as needed
  best_params = study.best_params
  model = A2C('MultiInputPolicy', env, verbose=1, **best_params, tensorboard_log="./tensorboard_logs/")
  '''
  # static hyperparameters
  model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.0005, ent_coef=0.9, tensorboard_log="./tensorboard_logs/") 
  model.learn(total_timesteps=60000)
  model.save("activation")
  print("Model Saved")
  del model

finetune=True
if(finetune):
  params = { 'learning_rate': 0.0003, 'n_steps': 1024, 'ent_coef': 0.1, 'batch_size': 128, 'n_epochs': 5 }
  model=PPO.load("activation", env, custom_objects=params)
  print(f"Hyper Params: lr:{model.learning_rate}; batch size:{model.batch_size}")
  model.learn(total_timesteps=80000)
  model.save("activation")
  print("Retrained model saved")
  del model

model=PPO.load("activation")
obs,_ = env.reset()
#testing, comment out for normal ops
testing=False
if(testing): 
  obs = _test_case1() 
  env.set_test_data(obs)

print("Reset obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["new_mdn_status"])

steps=0
for i in range(130):
    action, _state = model.predict(obs, deterministic=False)
    #action = env.action_space.sample()
    print("Doing ", Actions.get_action_type(action))
    obs, reward, done, terminated, info = env.step(action)
    steps +=1
    #print("episod ", i, "obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["mdn_status"],"reward:", reward, done)
    #print("mdn related to:", obs["mdn2sim"]," use existing? ",obs["use_existing_mdn"])
    env.render(mode="human")
    if done:
      print(f"Completed in {steps} steps")
      obs,_ = env.reset()
      steps=0
      #break

#env.step(env.action_space.sample())
