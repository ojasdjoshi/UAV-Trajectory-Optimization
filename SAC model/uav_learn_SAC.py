import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from uavenv_SAC import UavEnv
from energy_callback_SAC import EnergyCallback          # ← new callback

# --- Setup Directories ---
logdir     = "Energy Optimization/logs/uav_sac_optimized"
models_dir = f"Energy Optimization/models/uav_sac_optimized/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Initialize Environment ---
env = UavEnv()
env = Monitor(env, os.path.join(logdir, "monitor.csv"))
env.reset()

print(f"\n{'='*60}")
print(f"Environment Configuration:")
print(f"  Small-scale Fading : Enabled")
print(f"  Rician K-factor    : 10 dB (LOS links)")
print(f"  Shadowing Enabled  : Yes")
print(f"  Shadowing Std Dev  : 8.0 dB")
print(f"  Decorrelation Dist : 20.0 m")
print(f"{'='*60}\n")

# --- Initialize SAC Model ---
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=10000,
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[400, 300])
)

# --- Callback ---
energy_callback = EnergyCallback(verbose=1)

# --- Train ---
TOTAL_TIMESTEPS = 2_250_000
SAVE_INTERVAL   =   250_000

print(f"\n{'='*60}")
print(f"Starting SAC Training with Fading Channel Model")
print(f"Total Timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Max Episode Len : {env.unwrapped.max_steps} steps")
print(f"Save Interval   : {SAVE_INTERVAL:,} steps")
print(f"Model Directory : {models_dir}")
print(f"{'='*60}\n")

for checkpoint in range(SAVE_INTERVAL, TOTAL_TIMESTEPS + 1, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL,
        tb_log_name="SAC_Fading_Run",
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