import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import the environment we created previously
from uavenv import UavEnv

class SACMetricsCallback(BaseCallback):
    """
    Custom callback for logging UAV-specific metrics to TensorBoard for SAC.
    """
    def __init__(self, verbose=0):
        super(SACMetricsCallback, self).__init__(verbose)
        self.episodes_finished = 0
        self.curr_ep_rates = []
        self.curr_ep_uav_energies = []
        self.curr_ep_weighted_energies = []

    def _on_step(self) -> bool:
        # Extract info from the environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # Tracking data from uavenv.py info dictionary
            if "data_rate_kbps" in info:
                self.curr_ep_rates.append(info["data_rate_kbps"])
            if "uav_energy" in info:
                self.curr_ep_uav_energies.append(info["uav_energy"])
            if "weighted_energy" in info:
                self.curr_ep_weighted_energies.append(info["weighted_energy"])

        # Check if the episode ended
        if self.locals['dones'][0]:
            self.episodes_finished += 1
            
            # Calculate averages for the episode
            avg_rate = np.mean(self.curr_ep_rates) if self.curr_ep_rates else 0
            avg_uav_en = np.mean(self.curr_ep_uav_energies) if self.curr_ep_uav_energies else 0
            avg_weighted_en = np.mean(self.curr_ep_weighted_energies) if self.curr_ep_weighted_energies else 0
            
            # Record to TensorBoard
            self.logger.record("train/avg_episode_rate_kbps", avg_rate)
            self.logger.record("train/avg_episode_uav_energy", avg_uav_en)
            self.logger.record("train/avg_episode_weighted_energy", avg_weighted_en)
            
            # Reset buffers
            self.curr_ep_rates = []
            self.curr_ep_uav_energies = []
            self.curr_ep_weighted_energies = []
            
        return True

# --- Setup Directories ---
logdir = "logs/uav_sac_continuous"
models_dir = f"models/uav_sac_continuous/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
env = UavEnv()
env = Monitor(env, os.path.join(logdir, "monitor.csv"))

# --- Initialize SAC Model ---
# SAC is specifically designed for continuous action spaces
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    #device="cpu",           # FORCED CPU TRAINING
    tensorboard_log=logdir,
    learning_rate=3e-4,     # Standard LR for SAC
    buffer_size=1_000_000,  # Large replay buffer for off-policy learning
    learning_starts=1000,   # Steps of random actions before training starts
    batch_size=256,         # SAC usually prefers larger batches (e.g., 256)
    tau=0.005,              # Soft update coefficient for target networks
    gamma=0.99,             # Discount factor
    train_freq=1,           # Update the model after every step
    gradient_steps=1,       # Number of gradient steps per env step
    ent_coef='auto',        # Automatic entropy tuning (highly recommended)
    policy_kwargs=dict(
        net_arch=[256, 256] # Standard twin-Q and policy network architecture
    )
)

# --- Training Configuration ---
TOTAL_TIMESTEPS = 2_250_000  
SAVE_INTERVAL = 1_000_000

print(f"\n{'='*60}")
print(f"Starting SAC Training on CPU - Continuous 3D UAV Environment")
print(f"Algorithm: Soft Actor-Critic (SAC)")
print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
print(f"{'='*60}\n")

metrics_callback = SACMetricsCallback()

# Training loop with checkpoints
for checkpoint in range(SAVE_INTERVAL, TOTAL_TIMESTEPS + 1, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL,
        tb_log_name="SAC_UAV_Run",
        callback=metrics_callback,
        reset_num_timesteps=False
    )
    
    # Save checkpoint
    model.save(f"{models_dir}/sac_checkpoint_{checkpoint}")
    print(f"Checkpoint saved at {checkpoint} steps.")

# Save final model
model.save(f"{models_dir}/sac_final_model")
print(f"\nTraining Complete. Models saved to {models_dir}")
env.close()