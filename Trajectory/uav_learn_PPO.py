from stable_baselines3 import PPO
import os
import time
from uavenv import UavEnv
from stable_baselines3.common.monitor import Monitor
from data_rate_callback import DataRateCallback 

# --- Setup Directories ---
logdir = "logs/uav_ppo_continuous" 
models_dir = f"models/uav_ppo/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
env = UavEnv()
env = Monitor(env, logdir)
env.reset()

# --- Initialize PPO Model for Continuous Control ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0, # Policy entropy is handled differently in continuous spaces
)

# --- Initialize Custom Callback ---
data_rate_callback = DataRateCallback(verbose=1)

# --- Train the Model ---
TOTAL_TIMESTEPS = 2250000
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    tb_log_name="PPO_Continuous_SingleUser",
    callback=data_rate_callback 
)

# --- Save the Final Model ---
model.save(f"{models_dir}/final_model_continuous")
print(f"Model saved to {models_dir}")
env.close()