import os
import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from uavenv_RecPPO import UavEnv
from energy_callback_RecPPO import EnergyCallback          # ← new callback

# --- Setup Directories ---
logdir     = "Energy Optimization/logs/recurrent_ppo"
models_dir = f"Energy Optimization/models/recurrent_ppo/{int(time.time())}/"
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

# --- Initialize RecurrentPPO Model ---
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=2e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]),
        lstm_hidden_size=128,
        enable_critic_lstm=True
    )
)

# --- Callback ---
energy_callback = EnergyCallback(verbose=1)

# --- Train ---
TOTAL_TIMESTEPS = 2_250_000
SAVE_INTERVAL   =   250_000

print(f"\n{'='*60}")
print(f"Starting RecurrentPPO Training with Fading Channel Model")
print(f"Total Timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Max Episode Len : {env.unwrapped.max_steps} steps")
print(f"Save Interval   : {SAVE_INTERVAL:,} steps")
print(f"Model Directory : {models_dir}")
print(f"{'='*60}\n")

for checkpoint in range(SAVE_INTERVAL, TOTAL_TIMESTEPS + 1, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL,
        tb_log_name="RecurrentPPO_Fading_Run",
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