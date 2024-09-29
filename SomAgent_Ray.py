import numpy as np
import torch
import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from VZsampleEnv import MobilePhoneCarrierEnv, Actions
from gymnasium.envs.registration import register

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Ray
ray.shutdown()
ray.init()

# Register the custom environment
register(
    id='MobilePhoneCarrier-v0',
    entry_point='VZsampleEnv:MobilePhoneCarrierEnv',
)

# Define the env_creator function
def env_creator(env_config):
    return MobilePhoneCarrierEnv()

# Register the environment with Ray
register_env("MobilePhoneCarrier-v0", env_creator)

# Configuration
config = {
    "env": "MobilePhoneCarrier-v0",
    "num_workers": 4,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],
        "use_lstm": True,
        "lstm_cell_size": 256,
    },
    "lr": 0.005,
    "gamma": 0.99,
    "lambda": 0.95,
    "entropy_coeff": 0.9,
    "clip_param": 0.3,
    "num_sgd_iter": 10,
    "sgd_minibatch_size": 128,
    "train_batch_size": 4000,
}

def train_agent(config, num_iterations=10):
    try:
        # Create PPO trainer
        trainer = PPO(config=config)

        # Training loop
        for i in range(num_iterations):
            result = trainer.train()
            print(f"Iteration {i + 1}")
            
            # Print available metrics
            metrics = [
                "episode_reward_mean",
                "episode_len_mean",
                "episodes_this_iter",
                "training_iteration",
                "timesteps_total",
                "num_env_steps_sampled",
                "num_env_steps_trained",
            ]
            
            for metric in metrics:
                if metric in result:
                    print(f"  {metric}: {result[metric]}")
            
            # Print custom metrics if available
            if "custom_metrics" in result:
                print("  Custom Metrics:")
                for key, value in result["custom_metrics"].items():
                    print(f"    {key}: {value}")
            
            print("\n")  # Add a blank line for readability

        return trainer
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_agent(trainer, num_episodes=10):
    env = gym.make("MobilePhoneCarrier-v0")
    policy = trainer.get_policy()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        state = policy.get_initial_state()  # Get initial LSTM state
        
        while not done:
            action, state_out, _ = trainer.compute_single_action(
                observation=obs,
                state=state,
                explore=False,
                policy_id="default_policy"
            )
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            print(f"Step {step}: Action: {action}, Reward: {reward}")
            
            state = state_out  # Update LSTM state
        
        print(f"Episode {episode + 1} finished. Total Reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # Train the agent
    trained_trainer = train_agent(config)

    if trained_trainer is not None:
        # Save the trained model
        checkpoint_path = trained_trainer.save()
        print(f"Model saved at: {checkpoint_path}")

        # Evaluate the trained agent
        print("\nEvaluating trained agent:")
        evaluate_agent(trained_trainer)
    else:
        print("Training failed. Unable to evaluate or save the model.")

    # Clean up
    ray.shutdown()