from VZsampleEnv import MobilePhoneCarrierEnv, Actions
from stable_baselines3 import A2C, PPO

# Create an environment with a discrete action space of 1
env = MobilePhoneCarrierEnv()

# Access the action space
action_space = env.action_space

# The single possible action would typically have a value of 0
print(f"Action value:{action_space.sample()} \nObservations:{env.observation_space.sample()}")
print("\n Visualize obs:",env.observation_space)

#env.step(3)
#train model
model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.0005)
#model.learn(total_timesteps=50000)
#model.save("activation")

del model

model=PPO.load("activation")
obs,_ = env.reset()
# obs = FlattenObservation(obs)
print("Reset obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["mdn_status"])
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    print("Doing ", Actions.get_action_type(action))
    obs, reward, done, terminated, info = env.step(action)
    print("episod ", i, "obs:",obs["address_validation_status"],obs["device_validation_status"],obs["sim_validation_status"], obs["mdn_status"],"reward:", reward, done)
    #env.render()
    if done:
      obs,_ = env.reset()

#env.step(env.action_space.sample())