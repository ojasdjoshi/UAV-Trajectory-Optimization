import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
import numpy as np

# Import the environment from the file we just created
from uavenv import UavEnv

class EnergyCallback(BaseCallback):
    """
    Custom callback for logging specialized UAV metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(EnergyCallback, self).__init__(verbose)
        self.episodes_finished = 0
        self.curr_ep_rates = []
        self.curr_ep_energies = []

    def _on_step(self) -> bool:
        # 'infos' is a list of dictionaries from each environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            # Match the keys in uavenv.py
            if "data_rate_kbps" in info:
                self.curr_ep_rates.append(info["data_rate_kbps"])
            if "uav_energy" in info:
                self.curr_ep_energies.append(info["uav_energy"])

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episodes_finished += 1
            avg_rate = np.mean(self.curr_ep_rates) if self.curr_ep_rates else 0
            avg_energy = np.mean(self.curr_ep_energies) if self.curr_ep_energies else 0
            
            # Log to TensorBoard
            self.logger.record("train/avg_episode_rate_kbps", avg_rate)
            self.logger.record("train/avg_episode_uav_energy", avg_energy)
            
            # Reset for next episode
            self.curr_ep_rates = []
            self.curr_ep_energies = []
        return True

# --- Setup Directories ---
logdir = "logs/uav_ppo_continuous" 
models_dir = f"models/uav_ppo_continuous/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
env = UavEnv()
# Monitor needs a specific file path for the CSV log
env = Monitor(env, os.path.join(logdir, "monitor.csv"))

print(f"\n{'='*60}")
print(f"Continuous UAV Environment Initialized")
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print(f"{'='*60}\n")

# --- Initialize PPO Model ---
# Note: PPO automatically uses Gaussian policies for continuous Box spaces
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01, # Crucial for continuous exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
)

energy_callback = EnergyCallback()

# --- Training Configuration ---
TOTAL_TIMESTEPS = 2_250_000
SAVE_INTERVAL = 250_000

print(f"Starting Training: {TOTAL_TIMESTEPS} steps...")

for checkpoint in range(SAVE_INTERVAL, TOTAL_TIMESTEPS + 1, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL, 
        tb_log_name="PPO_Continuous_Run",
        callback=energy_callback,
        reset_num_timesteps=False
    )
    
    # Save checkpoint
    model.save(f"{models_dir}/checkpoint_{checkpoint}")
    print(f"Checkpoint saved: {checkpoint} steps")

# Final Save
model.save(f"{models_dir}/final_model")
print(f"\nTraining Complete. Logs at {logdir}")