import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import SAC  # Changed from PPO to SAC
import os
import math
import argparse

# Import the environment we created
from uavenv import UavEnv, SPACE_X, SPACE_Y, SPACE_Z, MIN_ALTITUDE, MAX_ALTITUDE

class UAVTrajectoryVisualizer:
    """
    Visualizes UAV trajectory in real-time with 2D and 3D views.
    Shows obstacles, user movement, performance metrics, velocity, and distance.
    Updated for SAC Algorithm, Continuous Action Space, and Variable Obstacle Heights.
    """
    
    def __init__(self, model_path, num_episodes=3, max_steps=200):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained SAC model
            num_episodes: Number of episodes to visualize
            max_steps: Maximum steps per episode
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Initialize environment
        self.env = UavEnv()
        
        # Load the trained model - Changed PPO.load to SAC.load
        print(f"Loading SAC model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
            
        self.model = SAC.load(model_path, env=self.env)
        
        # Storage for trajectory data
        self.reset_trajectory_data()
        
    def reset_trajectory_data(self):
        """Reset all trajectory tracking variables for a new episode."""
        self.uav_positions = []
        self.user_positions = []
        self.base_position = None
        self.obstacles = [] # Will store (x, y, height)
        self.data_rates = []
        self.rewards = []
        self.altitudes = []
        
        # Velocity and distance tracking
        self.speeds = []
        self.velocities = []
        self.step_distances = []
        self.cumulative_distances = []
        
    def collect_trajectory(self, episode_num=0):
        """
        Run one episode using the SAC model and collect trajectory data.
        
        Args:
            episode_num: Episode number for display
        """
        self.reset_trajectory_data()
        
        obs, info = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        running_distance = 0
        
        # Store initial positions from the environment
        self.base_position = (self.env.base.x, self.env.base.y, self.env.base.z)
        # Store (x, y, height) for each of the 9 obstacles
        self.obstacles = [(o.x, o.y, o.height) for o in self.env.obstacles]
        
        print(f"\n{'='*70}")
        print(f"   EPISODE {episode_num + 1}: SAC TRAJECTORY COLLECTION")
        print(f"{'='*70}")
        
        while not done and step < self.max_steps:
            # Predict the action from the trained SAC model
            # SAC deterministic=True is used for evaluation/visualization
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # UAV metrics calculation
            # Note: dt is assumed to be 1.0 (TIME_SLOT_DURATION)
            curr_v = self.env.uav.get_velocity(1.0) 
            step_dist = curr_v * 1.0
            running_distance += step_dist
            
            # Store trajectory data
            self.uav_positions.append((self.env.uav.x, self.env.uav.y, self.env.uav.z))
            self.user_positions.append((self.env.user.x, self.env.user.y, self.env.user.z))
            
            # Mapping keys from uavenv.py info dictionary
            self.data_rates.append(info.get('data_rate_kbps', 0))
            self.rewards.append(reward)
            self.altitudes.append(self.env.uav.z)
            
            # Calculate and store velocity and distance data
            self.speeds.append(curr_v)
            
            # Rough velocity vector estimation for 3D arrows
            vel_vec = np.array([
                self.env.uav.x - self.env.uav.prev_x,
                self.env.uav.y - self.env.uav.prev_y,
                self.env.uav.z - self.env.uav.prev_z
            ])
            self.velocities.append(vel_vec)
            self.step_distances.append(step_dist)
            self.cumulative_distances.append(running_distance)
            
            total_reward += reward
            step += 1
            
            # Display progress every 25 steps
            if step % 25 == 0:
                print(f"  Step {step:3d}/{self.max_steps} | "
                      f"Speed: {self.speeds[-1]:5.2f} m/s | "
                      f"Total Dist: {self.cumulative_distances[-1]:7.2f} m | "
                      f"Data Rate: {self.data_rates[-1]:5.2f} kbps")
        
        # Calculate statistics for the summary
        avg_speed = np.mean(self.speeds)
        max_speed = np.max(self.speeds)
        total_distance = self.cumulative_distances[-1] if self.cumulative_distances else 0
        avg_data_rate = np.mean(self.data_rates)
        
        print(f"\n{'─'*70}")
        print(f"   EPISODE {episode_num + 1} SUMMARY")
        print(f"{'─'*70}")
        print(f"   Total Steps:          {step}")
        print(f"   Total Reward:         {total_reward:.2f}")
        print(f"\n   DISTANCE & SPEED METRICS:")
        print(f"   ├─ Total Distance:    {total_distance:.2f} m")
        print(f"   ├─ Average Speed:     {avg_speed:.2f} m/s")
        print(f"   └─ Maximum Speed:     {max_speed:.2f} m/s")
        print(f"\n   DATA RATE METRICS:")
        print(f"   ├─ Average Data Rate: {avg_data_rate:.2f} kbps")
        print(f"   └─ Max Data Rate:     {np.max(self.data_rates):.2f} kbps")
        print(f"\n   STATUS:")
        print(f"   └─ Terminated Early:  {'YES (Crash/Goal) ❌' if terminated and step < self.max_steps else 'NO ✓'}")
        print(f"{'='*70}\n")
        
        return step
    
    def create_2d_animation(self, save_path="uav_trajectory_2d.gif", interval=100):
        """Create animated 2D top-down view of the mission."""
        print(f"  [→] Creating 2D animation...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- Top Left: Top-down view ---
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y)
        ax1.set_xlabel('X Position (m)'); ax1.set_ylabel('Y Position (m)')
        ax1.set_title('SAC Trajectory - Top View', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        ax1.plot(self.base_position[0], self.base_position[1], 'gs', markersize=15, 
                 label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        # Visualize all obstacles from the environment
        for idx, obs in enumerate(self.obstacles):
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.3, 
                            label='Obstacle' if idx == 0 else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'rx', markersize=10)
        
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue')
        user_point, = ax1.plot([], [], 'mo', markersize=10, label='User')
        ax1.legend(loc='upper right', fontsize=9)
        
        # --- Top Right: Data Rate & Altitude ---
        ax2.set_xlim(0, len(self.uav_positions))
        ax2.set_ylim(0, max(max(self.data_rates) * 1.1, 600))
        ax2.set_xlabel('Time Step'); ax2.set_ylabel('Data Rate (kbps)', color='blue')
        ax2.set_title('Data Rate & Altitude', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        data_rate_line, = ax2.plot([], [], 'b-', linewidth=2, label='Data Rate')
        
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylim(0, SPACE_Z)
        ax2_alt.set_ylabel('Altitude (m)', color='orange')
        ax2_alt.tick_params(axis='y', labelcolor='orange')
        altitude_line, = ax2_alt.plot([], [], 'orange', linewidth=2, label='Altitude')
        
        # --- Bottom Left: Speed ---
        ax3.set_xlim(0, len(self.uav_positions))
        ax3.set_ylim(0, max(max(self.speeds) * 1.1, 15))
        ax3.set_xlabel('Time Step'); ax3.set_ylabel('Speed (m/s)', color='green')
        ax3.set_title('UAV Speed', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        speed_line, = ax3.plot([], [], 'g-', linewidth=2)
        
        # --- Bottom Right: Cumulative Distance ---
        ax4.set_xlim(0, len(self.uav_positions))
        ax4.set_ylim(0, max(self.cumulative_distances) * 1.1 if self.cumulative_distances else 100)
        ax4.set_xlabel('Time Step'); ax4.set_ylabel('Distance (m)', color='purple')
        ax4.set_title('Cumulative Distance', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        distance_line, = ax4.plot([], [], 'purple', linewidth=2.5)
        
        # HUD overlay
        step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, 
                            verticalalignment='top', family='monospace', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        def init():
            uav_trail.set_data([], []); uav_point.set_data([], []); user_point.set_data([], [])
            data_rate_line.set_data([], []); altitude_line.set_data([], [])
            speed_line.set_data([], []); distance_line.set_data([], [])
            return uav_trail, uav_point, user_point, data_rate_line, altitude_line, speed_line, distance_line, step_text
        
        def animate(frame):
            x_trail = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_trail = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_trail, y_trail)
            uav_point.set_data([self.uav_positions[frame][0]], [self.uav_positions[frame][1]])
            user_point.set_data([self.user_positions[frame][0]], [self.user_positions[frame][1]])
            
            steps = list(range(frame + 1))
            data_rate_line.set_data(steps, self.data_rates[:frame+1])
            altitude_line.set_data(steps, self.altitudes[:frame+1])
            speed_line.set_data(steps, self.speeds[:frame+1])
            distance_line.set_data(steps, self.cumulative_distances[:frame+1])
            
            step_text.set_text(f'Step:     {frame + 1:3d}\n'
                               f'Speed:    {self.speeds[frame]:6.2f} m/s\n'
                               f'Total Dst:{self.cumulative_distances[frame]:7.2f} m\n'
                               f'Rate:     {self.data_rates[frame]:6.1f} kbps\n'
                               f'Alt:      {self.altitudes[frame]:6.1f} m')
            return uav_trail, uav_point, user_point, data_rate_line, altitude_line, speed_line, distance_line, step_text
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.uav_positions), interval=interval, blit=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"      ✓ 2D animation saved to: {save_path}")
    
    def create_3d_animation(self, save_path="uav_trajectory_3d.gif", interval=100):
        """Create animated 3D view with accurate obstacle heights."""
        print(f"  [→] Creating 3D animation...")
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, SPACE_X); ax.set_ylim(0, SPACE_Y); ax.set_zlim(0, SPACE_Z)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('SAC 3D Trajectory - Continuous Control', fontsize=14, fontweight='bold')
        
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                   c='green', marker='s', s=200, label='Base Station')
        
        # Render obstacles pull from data list (includes unique heights)
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, obs[2], 2) # Vertical scale based on unique height
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.15, color='red')
        
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7, label='UAV Path')
        uav_point = ax.scatter([], [], [], c='blue', marker='o', s=150)
        user_point = ax.scatter([], [], [], c='magenta', marker='o', s=100, label='User')
        
        velocity_arrow = None
        step_text = fig.text(0.02, 0.95, '', fontsize=10, family='monospace', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        def animate(frame):
            nonlocal velocity_arrow
            x_t = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_t = [pos[1] for pos in self.uav_positions[:frame+1]]
            z_t = [pos[2] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_t, y_t); uav_trail.set_3d_properties(z_t)
            uav_point._offsets3d = ([self.uav_positions[frame][0]], [self.uav_positions[frame][1]], [self.uav_positions[frame][2]])
            user_point._offsets3d = ([self.user_positions[frame][0]], [self.user_positions[frame][1]], [self.user_positions[frame][2]])
            
            # Quiver velocity vector
            if velocity_arrow: velocity_arrow.remove()
            vel = self.velocities[frame]
            velocity_arrow = ax.quiver(self.uav_positions[frame][0], self.uav_positions[frame][1], self.uav_positions[frame][2],
                                        vel[0]*3, vel[1]*3, vel[2]*3, color='red', alpha=0.6)
            
            step_text.set_text(f'Step:      {frame + 1:3d}\n'
                               f'Speed:     {self.speeds[frame]:6.2f} m/s\n'
                               f'Total Dst: {self.cumulative_distances[frame]:7.2f} m\n'
                               f'Rate:      {self.data_rates[frame]:6.1f} kbps')
            ax.view_init(elev=20, azim=45 + frame * 0.3)
            return uav_trail, uav_point, user_point, step_text
        
        anim = FuncAnimation(fig, animate, frames=len(self.uav_positions), interval=interval)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"      ✓ 3D animation saved to: {save_path}")
    
    def create_static_plot(self, save_path="uav_trajectory_static.png"):
        """Create a multi-panel static summary plot of the episode."""
        print(f"  [→] Creating static trajectory plot...")
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 3D Overview
        ax1 = fig.add_subplot(3, 2, 1, projection='3d')
        ax1.set_title('3D Trajectory Overview (SAC)', fontweight='bold')
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y); ax1.set_zlim(0, SPACE_Z)
        
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, obs[2], 2)
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.1, color='red')
            
        uav_x = [p[0] for p in self.uav_positions]
        uav_y = [p[1] for p in self.uav_positions]
        uav_z = [p[2] for p in self.uav_positions]
        ax1.plot(uav_x, uav_y, uav_z, 'b-', alpha=0.7, label='UAV Path')
        ax1.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                    c='green', marker='s', s=100)
        
        # 2. Top Down View
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.set_title('Top-Down Trajectory Map', fontweight='bold')
        ax2.set_xlim(0, SPACE_X); ax2.set_ylim(0, SPACE_Y); ax2.set_aspect('equal')
        for obs in self.obstacles:
            ax2.add_patch(Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.2))
        ax2.plot(uav_x, uav_y, 'b-', alpha=0.6)
        ax2.scatter([self.base_position[0]], [self.base_position[1]], c='green', marker='s')
        
        # 3. Data Rate Plot
        steps = range(len(self.uav_positions))
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(steps, self.data_rates, 'b-'); ax3.set_title('Data Rate (kbps)'); ax3.grid(True)
        
        # 4. Speed Plot
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(steps, self.speeds, 'g-'); ax4.set_title('UAV Speed (m/s)'); ax4.grid(True)
        
        # 5. Distance Plot
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(steps, self.cumulative_distances, 'purple'); ax5.set_title('Total Distance (m)'); ax5.grid(True)
        
        # 6. Statistics Summary
        ax6 = fig.add_subplot(3, 2, 6); ax6.axis('off')
        stats = (f'SAC MISSION STATISTICS\n'
                 f'Total Distance: {self.cumulative_distances[-1]:.1f}m\n'
                 f'Avg Speed:      {np.mean(self.speeds):.2f}m/s\n'
                 f'Avg Altitude:   {np.mean(self.altitudes):.1f}m\n'
                 f'Avg Rate:       {np.mean(self.data_rates):.1f}kbps\n'
                 f'Total Reward:   {np.sum(self.rewards):.1f}')
        ax6.text(0.1, 0.8, stats, fontsize=14, family='monospace', bbox=dict(facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"      ✓ Static plot saved to: {save_path}")
    
    def visualize_all(self, output_dir="Energy Optimization/visualizations"):
        """Run full visualization suite."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n╔{'═'*68}╗")
        print(f"║   UAV SAC TRAJECTORY VISUALIZATION SYSTEM                       ║")
        print(f"╚{'═'*68}╝\n")
        
        for episode in range(self.num_episodes):
            self.collect_trajectory(episode)
            suffix = f"_ep{episode+1}" if self.num_episodes > 1 else ""
            self.create_static_plot(os.path.join(output_dir, f"static{suffix}.png"))
            self.create_2d_animation(os.path.join(output_dir, f"2d_anim{suffix}.gif"))
            self.create_3d_animation(os.path.join(output_dir, f"3d_anim{suffix}.gif"))
        
        print(f"\n✓ All SAC Visualizations complete in '{output_dir}/'")
        self.env.close()

if __name__ == "__main__":
    # Ensure this path matches your trained SAC model file
    MODEL_PATH = r"Energy Optimization\models\uav_sac_optimized\1770923807\sac_final_model.zip"
    
    parser = argparse.ArgumentParser(description='UAV SAC visualizer')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to SAC model .zip file')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Number of episodes to visualize')
    args = parser.parse_args()

    if os.path.exists(args.model):
        visualizer = UAVTrajectoryVisualizer(model_path=args.model, num_episodes=args.episodes)
        visualizer.visualize_all()
    else:
        print(f"ERROR: SAC Model not found at {MODEL_PATH}.")
        print("Please verify the directory path in the main block of the script.")