import os
import time
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO  # Using sb3-contrib for LSTM support
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from uavenv_RecPPO import UavEnv

class PPOMetricsCallback(BaseCallback):
    """
    Custom callback for logging 3GPP specific metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(PPOMetricsCallback, self).__init__(verbose)
        self.curr_ep_rates = []
        self.curr_ep_uav_energies = []

    def _on_step(self) -> bool:
        # SB3-Contrib stores info in a list for vectorized envs
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if "data_rate_kbps" in info: 
                self.curr_ep_rates.append(info["data_rate_kbps"])
            if "uav_energy" in info: 
                self.curr_ep_uav_energies.append(info["uav_energy"])

        # Check if the episode ended
        if self.locals['dones'][0]:
            if self.curr_ep_rates:
                avg_rate = np.mean(self.curr_ep_rates)
                avg_uav_en = np.mean(self.curr_ep_uav_energies)
                self.logger.record("train/avg_episode_rate_kbps", avg_rate)
                self.logger.record("train/avg_episode_uav_energy", avg_uav_en)
            
            # Reset trackers for next episode
            self.curr_ep_rates, self.curr_ep_uav_energies = [], []
        return True

# --- Path Configuration ---
logdir = "Energy Optimization/logs/recurrent_ppo"
models_dir = f"Energy Optimization/models/recurrent_ppo/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Environment Setup ---
env = UavEnv()
env = Monitor(env, os.path.join(logdir, "monitor.csv"))

# --- Recurrent PPO Model Configuration ---
# Recurrent PPO is on-policy, so it doesn't use a 'buffer_size' like SAC.
# It uses 'n_steps' to define the sequence length for the LSTM.
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=2e-4,       # Slightly higher than SAC but stable for LSTM
    n_steps=512,              # Collected steps per update (covers ~2.5 full episodes)
    batch_size=64,            # Number of sequences to process in parallel
    n_epochs=10,              # How many times to optimize per rollout
    gamma=0.99,
    gae_lambda=0.95,          # Standard for PPO stability
    clip_range=0.2,           # Prevents policy from changing too drastically
    ent_coef=0.01,            # Encourages exploration of new paths around cylinders
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]), # Shared layers before LSTM
        lstm_hidden_size=128,
        enable_critic_lstm=True # Allows the value function to also have memory
    )
)



# --- Training Execution ---
TOTAL_TIMESTEPS = 2_000_000  # PPO often converges faster than SAC in discrete-like steps
metrics_callback = PPOMetricsCallback()

print(f"Starting training for {TOTAL_TIMESTEPS} steps...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="Recurrent_PPO_3GPP_Run",
    callback=metrics_callback
)

# Save the memory-enabled model
model.save(f"{models_dir}/recurrent_ppo_final_model")
print(f"Model saved to {models_dir}")

env.close()