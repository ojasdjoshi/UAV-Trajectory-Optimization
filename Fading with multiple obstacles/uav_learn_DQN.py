import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from uavenv_discrete import UavEnv
# Ensure your callback file is named correctly or adjust the import
try:
    from data_rate_callback import DataRateCallback 
except ImportError:
    # Fallback if the callback file isn't found
    DataRateCallback = None

# --- Setup Directories ---
logdir = "logs/uav_dqn_discrete" 
models_dir = f"models/uav_dqn/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
# Using the discrete environment generated in uav_env_discrete.py
env = UavEnv()
env = Monitor(env, logdir)
env.reset()

# --- Initialize DQN Model ---
# DQN is designed for discrete action spaces like the one we just created
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=1e-4,          # Often lower for DQN stability
    buffer_size=100000,          # Size of replay buffer
    learning_starts=1000,        # How many steps to take before starting to learn
    batch_size=32,               # Number of samples per gradient update
    tau=1.0,                     # Soft update coefficient (1.0 = hard update)
    gamma=0.99,                  # Discount factor
    train_freq=4,                # Update the model every 4 steps
    gradient_steps=1,            # How many gradient steps to take after each rollout
    target_update_interval=1000, # Update target network every 1000 steps
    exploration_fraction=0.1,    # Fraction of total steps for epsilon decay
    exploration_final_eps=0.05,  # Final value of random action probability
)

# --- Initialize Custom Callback ---
callbacks = []
if DataRateCallback:
    data_rate_callback = DataRateCallback(verbose=1)
    callbacks.append(data_rate_callback)

# --- Train the Model ---
TOTAL_TIMESTEPS = 2250000
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    tb_log_name="DQN_Discrete_SingleUser",
    callback=callbacks if callbacks else None
)

# --- Save the Final Model ---
model.save(f"{models_dir}/final_model_discrete")
print(f"DQN Model saved to {models_dir}")
env.close()