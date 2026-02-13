import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from uavenv import UavEnv

class SACMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SACMetricsCallback, self).__init__(verbose)
        self.curr_ep_rates = []
        self.curr_ep_uav_energies = []

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if "data_rate_kbps" in info: self.curr_ep_rates.append(info["data_rate_kbps"])
            if "uav_energy" in info: self.curr_ep_uav_energies.append(info["uav_energy"])

        if self.locals['dones'][0]:
            avg_rate = np.mean(self.curr_ep_rates) if self.curr_ep_rates else 0
            avg_uav_en = np.mean(self.curr_ep_uav_energies) if self.curr_ep_uav_energies else 0
            self.logger.record("train/avg_episode_rate_kbps", avg_rate)
            self.logger.record("train/avg_episode_uav_energy", avg_uav_en)
            self.curr_ep_rates, self.curr_ep_uav_energies = [], []
        return True

logdir = "Energy Optimization/logs/uav_sac_optimized"
models_dir = f"Energy Optimization/models/uav_sac_optimized/{int(time.time())}/"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

env = UavEnv()
env = Monitor(env, os.path.join(logdir, "monitor.csv"))

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=1e-4,       # LOWER: Prevents the "weird" spikes and dips
    buffer_size=1_000_000,
    learning_starts=10000,    # HIGHER: Agent collects more random data before learning
    batch_size=512,           # LARGER: More stable gradients in noisy fading channels
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',          # Still auto, but lr=1e-4 will stabilize it
    policy_kwargs=dict(net_arch=[400, 300]) # Slightly asymmetrical for better feature extraction
)

TOTAL_TIMESTEPS = 2_250_000
metrics_callback = SACMetricsCallback()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="SAC_Optimized_Run",
    callback=metrics_callback
)

model.save(f"{models_dir}/sac_final_model")
env.close()