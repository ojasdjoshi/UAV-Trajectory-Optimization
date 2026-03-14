import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from uavenv import UavEnv
from energy_callback_PPO import EnergyCallback          # ← separate callback file

# --- Setup Directories ---
logdir     = "logs/uav_ppo_continuous"
models_dir = f"models/uav_ppo_continuous/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
env = UavEnv()
env = Monitor(env, os.path.join(logdir, "monitor.csv"))
env.reset()

print(f"\n{'='*60}")
print(f"Continuous UAV Environment Initialized")
print(f"Action Space      : {env.action_space}")
print(f"Observation Space : {env.observation_space}")
print(f"{'='*60}\n")

# --- Initialize PPO Model ---
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
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
)

# --- Callback ---
energy_callback = EnergyCallback(verbose=1)

# --- Training Configuration ---
TOTAL_TIMESTEPS = 2_250_000
SAVE_INTERVAL   =   250_000

print(f"\n{'='*60}")
print(f"Starting PPO Training with Fading Channel Model")
print(f"Total Timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Max Episode Len : {env.unwrapped.max_steps} steps")
print(f"Save Interval   : {SAVE_INTERVAL:,} steps")
print(f"Model Directory : {models_dir}")
print(f"{'='*60}\n")

for checkpoint in range(SAVE_INTERVAL, TOTAL_TIMESTEPS + 1, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL,
        tb_log_name="PPO_Continuous_Run",
        callback=energy_callback,
        reset_num_timesteps=False,
    )
    model.save(f"{models_dir}/checkpoint_{checkpoint}")
    print(f"\nCheckpoint saved at {checkpoint:,} steps")

# --- Final Save ---
model.save(f"{models_dir}/final_model_{TOTAL_TIMESTEPS}")

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Model saved to     : {models_dir}")
print(f"TensorBoard logs   : {logdir}")
print(f"Total Episodes     : {energy_callback.episodes_finished}")
print(f"{'='*60}\n")
print(f"To view training progress, run:")
print(f"tensorboard --logdir={logdir}")

env.close()