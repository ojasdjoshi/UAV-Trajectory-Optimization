from stable_baselines3 import PPO
from uavenv import UavEnv
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Configuration ---
# IMPORTANT: Update this path to your saved model
# Use a Raw String (r"...") to avoid SyntaxError on Windows paths
model_path = r"C:\Users\ASUS\OneDrive\Desktop\fyp RL\data rate model\models\uav_ppo\1761479906\final_model_1200000.zip"
episodes = 10 # Number of evaluation episodes

# --- Initialize Environment and Load Model ---
env = UavEnv()
try:
    model = PPO.load(model_path, env=env)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model path is correct and the .zip file exists.")
    exit()

# --- Evaluation and Data Collection ---
all_timesteps_rates = []
current_total_steps = 0

print("Starting evaluation...")

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    episode_rate_data = []
    
    while not done:
        # Predict the action
        action, _ = model.predict(obs, deterministic=True)
        # Take a step and get the reward (which is the instantaneous data rate)
        obs, reward, done, _, _ = env.step(action)
        
        # Log the instantaneous rate and the global step count
        all_timesteps_rates.append(reward)
        current_total_steps += 1
        
    print(f"EPISODE {ep + 1} finished (Steps: {current_total_steps - len(episode_rate_data)})")

# --- Plotting Function ---
def plot_data_rate_curve(rates_data):
    """Plots the instantaneous data rate against all timesteps."""
    
    # Generate the x-axis (timesteps)
    timesteps = np.arange(1, len(rates_data) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot the instantaneous rate
    plt.plot(timesteps, rates_data, alpha=0.6, label='Instantaneous Data Rate (Reward)', linewidth=1)

    # 2. Calculate and plot a rolling mean (smoother curve)
    # Use a window size appropriate for the max episode length (200)
    window_size = 50 
    rolling_mean = np.convolve(rates_data, np.ones(window_size)/window_size, mode='valid')
    # The rolling mean is shorter, so adjust the x-axis for plotting
    plt.plot(timesteps[window_size - 1:], rolling_mean, color='red', label=f'Rolling Mean (Window {window_size})', linewidth=2)
    
    plt.title(f"Data Rate (Reward) over {len(rates_data)} Evaluation Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Bottleneck Data Rate (bits/Hz)")
    plt.legend()
    plt.grid(axis='both', linestyle='--')
    plt.ylim(bottom=0) # Data rate cannot be negative
    plt.show()

# --- Execute Plotting ---
plot_data_rate_curve(all_timesteps_rates)

env.close()