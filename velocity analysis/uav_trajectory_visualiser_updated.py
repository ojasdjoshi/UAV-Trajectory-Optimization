import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from uavenv import UavEnv, SPACE_X, SPACE_Y, SPACE_Z, MIN_ALTITUDE, MAX_ALTITUDE
import os

class UAVTrajectoryVisualizer:
    """
    Visualizes UAV trajectory in real-time with 2D and 3D views.
    Shows 5 obstacles with varying heights, user movement, performance metrics, velocity, and distance.
    """
    
    def __init__(self, model_path, num_episodes=1, max_steps=200):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained PPO model
            num_episodes: Number of episodes to visualize
            max_steps: Maximum steps per episode
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Initialize environment
        self.env = UavEnv()
        
        # Load the trained model - passing env ensures space consistency
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, env=self.env)
        
        # Storage for trajectory data
        self.reset_trajectory_data()
        
    def reset_trajectory_data(self):
        """Reset all trajectory tracking variables."""
        self.uav_positions = []
        self.user_positions = []
        self.base_position = None
        self.obstacles = []
        self.obstacle_heights = []  # NEW: Store obstacle heights
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
        Run one episode and collect trajectory data.
        
        Args:
            episode_num: Episode number for display
        """
        self.reset_trajectory_data()
        
        obs, info = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Store initial positions
        self.base_position = (self.env.base.x, self.env.base.y, self.env.base.z)
        self.obstacles = [(o.x, o.y, o.z) for o in self.env.obstacles]
        self.obstacle_heights = [o.z for o in self.env.obstacles]  # Store heights
        
        print(f"\n{'='*70}")
        print(f"  EPISODE {episode_num + 1}: TRAJECTORY COLLECTION")
        print(f"{'='*70}")
        print(f"  Environment Configuration:")
        print(f"  ├─ Number of Obstacles: {len(self.obstacles)}")
        print(f"  ├─ Obstacle Heights: {[f'{h}m' for h in self.obstacle_heights]}")
        print(f"  └─ User Movement: Active (every 10 steps)")
        print(f"{'─'*70}")
        
        while not done and step < self.max_steps:
            # Predict the action from the trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            self.uav_positions.append((self.env.uav.x, self.env.uav.y, self.env.uav.z))
            self.user_positions.append((self.env.user.x, self.env.user.y, self.env.user.z))
            self.data_rates.append(info.get('data_rate', 0))
            self.rewards.append(reward)
            self.altitudes.append(self.env.uav.z)
            
            # Store velocity and distance data
            self.speeds.append(info.get('uav_speed', 0))
            self.velocities.append(info.get('uav_velocity', np.array([0, 0, 0])))
            self.step_distances.append(info.get('step_distance', 0))
            self.cumulative_distances.append(info.get('total_distance', 0))
            
            total_reward += reward
            step += 1
            
            # Display progress every 25 steps with speed and distance
            if step % 25 == 0:
                print(f"  Step {step:3d}/{self.max_steps} | "
                      f"Speed: {self.speeds[-1]:5.2f} m/s | "
                      f"Step Dist: {self.step_distances[-1]:5.2f} m | "
                      f"Total Dist: {self.cumulative_distances[-1]:7.2f} m | "
                      f"Data Rate: {self.data_rates[-1]:5.2f} bps/Hz")
        
        # Calculate statistics
        avg_speed = np.mean(self.speeds)
        max_speed = np.max(self.speeds)
        min_speed = np.min(self.speeds)
        total_distance = self.cumulative_distances[-1] if self.cumulative_distances else 0
        avg_data_rate = np.mean(self.data_rates)
        
        # Calculate user movement
        user_distances = []
        for i in range(len(self.user_positions) - 1):
            dx = self.user_positions[i+1][0] - self.user_positions[i][0]
            dy = self.user_positions[i+1][1] - self.user_positions[i][1]
            user_distances.append(np.sqrt(dx**2 + dy**2))
        total_user_movement = sum(user_distances)
        
        print(f"\n{'─'*70}")
        print(f"  EPISODE {episode_num + 1} SUMMARY")
        print(f"{'─'*70}")
        print(f"  Total Steps:          {step}")
        print(f"  Total Reward:         {total_reward:.2f}")
        print(f"\n  UAV DISTANCE & SPEED METRICS:")
        print(f"  ├─ Total Distance:    {total_distance:.2f} m")
        print(f"  ├─ Average Speed:     {avg_speed:.2f} m/s")
        print(f"  ├─ Maximum Speed:     {max_speed:.2f} m/s")
        print(f"  └─ Minimum Speed:     {min_speed:.2f} m/s")
        print(f"\n  USER MOVEMENT METRICS:")
        print(f"  ├─ Total Movement:    {total_user_movement:.2f} m")
        print(f"  ├─ Movement Steps:    {len([d for d in user_distances if d > 0])}")
        print(f"  └─ Avg Move Distance: {(total_user_movement/max(1, len([d for d in user_distances if d > 0]))):.2f} m/step")
        print(f"\n  DATA RATE METRICS:")
        print(f"  ├─ Average Data Rate: {avg_data_rate:.2f} bps/Hz")
        print(f"  ├─ Max Data Rate:     {np.max(self.data_rates):.2f} bps/Hz")
        print(f"  └─ Min Data Rate:     {np.min(self.data_rates):.2f} bps/Hz")
        print(f"\n  ALTITUDE METRICS:")
        print(f"  ├─ Average Altitude:  {np.mean(self.altitudes):.1f} m")
        print(f"  ├─ Max Altitude:      {np.max(self.altitudes):.1f} m")
        print(f"  └─ Min Altitude:      {np.min(self.altitudes):.1f} m")
        print(f"\n  STATUS:")
        print(f"  └─ Crashed:           {'YES ❌' if terminated and step < self.max_steps else 'NO ✓'}")
        print(f"{'='*70}\n")
        
        return step
    
    def create_2d_animation(self, save_path="uav_trajectory_2d.gif", interval=100):
        """
        Create animated 2D top-down view of the trajectory with 5 obstacles and user movement.
        """
        print(f"  [→] Creating 2D animation...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- Top Left: Top-down view ---
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_xlabel('X Position (m)', fontsize=11)
        ax1.set_ylabel('Y Position (m)', fontsize=11)
        ax1.set_title('UAV Trajectory - Top View (5 Obstacles)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot static elements
        ax1.plot(self.base_position[0], self.base_position[1], 
                'gs', markersize=15, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        # Plot 5 obstacles with varying colors based on height
        obstacle_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(self.obstacles)))
        for idx, (obs, height) in enumerate(zip(self.obstacles, self.obstacle_heights)):
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color=obstacle_colors[idx], alpha=0.4, 
                          label=f'Obs {idx+1} ({height}m)' if idx < 5 else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'kx', markersize=10, markeredgewidth=2)
            # Add height label
            ax1.text(obs[0], obs[1], f'{height}m', fontsize=8, ha='center', 
                    va='center', fontweight='bold', color='darkred')
        
        # Initialize dynamic elements
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        user_trail, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.5, label='User Path')
        user_point, = ax1.plot([], [], 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        
        ax1.legend(loc='upper right', fontsize=9)
        
        # --- Top Right: Data Rate ---
        ax2.set_xlim(0, len(self.data_rates))
        ax2.set_ylim(0, max(self.data_rates) * 1.2 if self.data_rates else 10)
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Data Rate (bps/Hz)', fontsize=11)
        ax2.set_title('Data Rate Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        rate_line, = ax2.plot([], [], 'b-', linewidth=2)
        rate_point, = ax2.plot([], [], 'bo', markersize=8)
        
        # --- Bottom Left: Altitude & Speed ---
        ax3.set_xlim(0, len(self.altitudes))
        ax3.set_ylim(MIN_ALTITUDE - 10, MAX_ALTITUDE + 10)
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Altitude (m)', fontsize=11, color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        ax3.set_title('Altitude & Speed Over Time', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        alt_line, = ax3.plot([], [], 'orange', linewidth=2, label='Altitude')
        alt_point, = ax3.plot([], [], 'o', color='orange', markersize=8)
        
        # Draw obstacle height lines
        for idx, height in enumerate(self.obstacle_heights):
            ax3.axhline(y=height, color=obstacle_colors[idx], linestyle='--', 
                       alpha=0.5, linewidth=1.5, label=f'Obs {idx+1} Height')
        
        ax3_speed = ax3.twinx()
        ax3_speed.set_ylabel('Speed (m/s)', fontsize=11, color='green')
        ax3_speed.tick_params(axis='y', labelcolor='green')
        ax3_speed.set_ylim(0, max(self.speeds) * 1.2 if self.speeds else 10)
        
        speed_line, = ax3_speed.plot([], [], 'green', linewidth=2, label='Speed')
        speed_point, = ax3_speed.plot([], [], 'go', markersize=8)
        
        # Combine legends
        lines_3 = [alt_line, speed_line]
        labels_3 = ['Altitude', 'Speed']
        ax3.legend(lines_3, labels_3, loc='upper left', fontsize=9)
        
        # --- Bottom Right: Info Panel ---
        ax4.axis('off')
        ax4.set_title('Real-time Statistics', fontsize=13, fontweight='bold')
        info_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, 
                            fontsize=10, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        def init():
            uav_trail.set_data([], [])
            uav_point.set_data([], [])
            user_trail.set_data([], [])
            user_point.set_data([], [])
            rate_line.set_data([], [])
            rate_point.set_data([], [])
            alt_line.set_data([], [])
            alt_point.set_data([], [])
            speed_line.set_data([], [])
            speed_point.set_data([], [])
            return (uav_trail, uav_point, user_trail, user_point, 
                   rate_line, rate_point, alt_line, alt_point, 
                   speed_line, speed_point, info_text)
        
        def animate(frame):
            # UAV trajectory
            uav_x = [pos[0] for pos in self.uav_positions[:frame+1]]
            uav_y = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(uav_x, uav_y)
            uav_point.set_data([uav_x[-1]], [uav_y[-1]])
            
            # User trajectory
            user_x = [pos[0] for pos in self.user_positions[:frame+1]]
            user_y = [pos[1] for pos in self.user_positions[:frame+1]]
            user_trail.set_data(user_x, user_y)
            user_point.set_data([user_x[-1]], [user_y[-1]])
            
            # Data rate
            steps = list(range(frame + 1))
            rate_line.set_data(steps, self.data_rates[:frame+1])
            rate_point.set_data([frame], [self.data_rates[frame]])
            
            # Altitude
            alt_line.set_data(steps, self.altitudes[:frame+1])
            alt_point.set_data([frame], [self.altitudes[frame]])
            
            # Speed
            speed_line.set_data(steps, self.speeds[:frame+1])
            speed_point.set_data([frame], [self.speeds[frame]])
            
            # Info panel
            info_text.set_text(
                f'╔═══════════════════════════════════╗\n'
                f'║   REAL-TIME METRICS               ║\n'
                f'╚═══════════════════════════════════╝\n'
                f'Step:         {frame + 1:3d}/{len(self.uav_positions)}\n'
                f'─────────────────────────────────────\n'
                f'📍 UAV Position:\n'
                f'  X: {self.uav_positions[frame][0]:6.1f} m\n'
                f'  Y: {self.uav_positions[frame][1]:6.1f} m\n'
                f'  Z: {self.uav_positions[frame][2]:6.1f} m\n'
                f'─────────────────────────────────────\n'
                f'👤 User Position:\n'
                f'  X: {self.user_positions[frame][0]:6.1f} m\n'
                f'  Y: {self.user_positions[frame][1]:6.1f} m\n'
                f'─────────────────────────────────────\n'
                f'⚡ UAV Metrics:\n'
                f'  Speed:      {self.speeds[frame]:6.2f} m/s\n'
                f'  Distance:   {self.cumulative_distances[frame]:6.2f} m\n'
                f'  Data Rate:  {self.data_rates[frame]:6.2f} bps/Hz\n'
                f'─────────────────────────────────────\n'
                f'🏢 Obstacles: {len(self.obstacles)} buildings\n'
                f'  Heights: {min(self.obstacle_heights)}-{max(self.obstacle_heights)}m'
            )
            
            return (uav_trail, uav_point, user_trail, user_point, 
                   rate_line, rate_point, alt_line, alt_point, 
                   speed_line, speed_point, info_text)
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=False, repeat=True)
        
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"      ✓ 2D animation saved to: {save_path}")
        
        plt.close()
    
    def create_3d_animation(self, save_path="uav_trajectory_3d.gif", interval=100):
        """
        Create animated 3D view with 5 obstacles (cylinders with varying heights) and user movement.
        """
        print(f"  [→] Creating 3D animation...")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, SPACE_X)
        ax.set_ylim(0, SPACE_Y)
        ax.set_zlim(0, SPACE_Z)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title('3D UAV Trajectory with 5 Obstacles & User Movement', fontsize=14, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])
        
        # Base station
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                  c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        # Plot 5 cylindrical obstacles with varying heights
        obstacle_colors_3d = plt.cm.Reds(np.linspace(0.4, 0.9, len(self.obstacles)))
        for idx, (obs, height) in enumerate(zip(self.obstacles, self.obstacle_heights)):
            # Create cylinder for obstacle
            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(0, height, 20)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = obs[0] + self.env.obstacle_radius * np.cos(theta_grid)
            y_cyl = obs[1] + self.env.obstacle_radius * np.sin(theta_grid)
            
            ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color=obstacle_colors_3d[idx],
                          label=f'Obstacle {idx+1} ({height}m)' if idx == 0 else '')
            
            # Add top circle
            x_top = obs[0] + self.env.obstacle_radius * np.cos(theta)
            y_top = obs[1] + self.env.obstacle_radius * np.sin(theta)
            z_top = np.full_like(theta, height)
            ax.plot(x_top, y_top, z_top, color=obstacle_colors_3d[idx], linewidth=2)
            
            # Label
            ax.text(obs[0], obs[1], height + 5, f'Obs{idx+1}\n{height}m', 
                   fontsize=8, ha='center', color='darkred', fontweight='bold')
        
        # Initialize trajectory lines
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7, label='UAV Path')
        uav_point, = ax.plot([], [], [], 'bo', markersize=10, markeredgecolor='darkblue', 
                            markeredgewidth=2, label='UAV Current')
        user_trail, = ax.plot([], [], [], 'r--', linewidth=2, alpha=0.5, label='User Path')
        user_point, = ax.plot([], [], [], 'ro', markersize=8, markeredgecolor='darkred', 
                             markeredgewidth=2, label='User Current')
        
        ax.legend(loc='upper left', fontsize=9)
        
        # Info text
        step_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, 
                             verticalalignment='top', family='monospace',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        def init():
            uav_trail.set_data([], [])
            uav_trail.set_3d_properties([])
            uav_point.set_data([], [])
            uav_point.set_3d_properties([])
            user_trail.set_data([], [])
            user_trail.set_3d_properties([])
            user_point.set_data([], [])
            user_point.set_3d_properties([])
            return uav_trail, uav_point, user_trail, user_point, step_text
        
        def animate(frame):
            # UAV trail
            uav_x = [pos[0] for pos in self.uav_positions[:frame+1]]
            uav_y = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_z = [pos[2] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(uav_x, uav_y)
            uav_trail.set_3d_properties(uav_z)
            uav_point.set_data([uav_x[-1]], [uav_y[-1]])
            uav_point.set_3d_properties([uav_z[-1]])
            
            # User trail
            user_x = [pos[0] for pos in self.user_positions[:frame+1]]
            user_y = [pos[1] for pos in self.user_positions[:frame+1]]
            user_z = [pos[2] for pos in self.user_positions[:frame+1]]
            user_trail.set_data(user_x, user_y)
            user_trail.set_3d_properties(user_z)
            user_point.set_data([user_x[-1]], [user_y[-1]])
            user_point.set_3d_properties([user_z[-1]])
            
            # Update info text
            step_text.set_text(
                f'Step:      {frame + 1:3d}/{len(self.uav_positions)}\n'
                f'Speed:     {self.speeds[frame]:6.2f} m/s\n'
                f'Distance:  {self.cumulative_distances[frame]:7.2f} m\n'
                f'Data Rate: {self.data_rates[frame]:6.2f} bps/Hz\n'
                f'Altitude:  {self.altitudes[frame]:6.1f} m\n'
                f'Obstacles: {len(self.obstacles)} buildings'
            )
            
            # Rotate view
            ax.view_init(elev=20, azim=45 + frame * 0.5)
            
            return uav_trail, uav_point, user_trail, user_point, step_text
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=False, repeat=True)
        
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"      ✓ 3D animation saved to: {save_path}")
        
        plt.close()
    
    def create_static_plot(self, save_path="uav_trajectory_static.png"):
        """
        Create a comprehensive static plot with multiple views including 5 obstacles, 
        user movement, velocity and distance.
        """
        print(f"  [→] Creating static trajectory plot...")
        
        fig = plt.figure(figsize=(22, 16))
        
        # --- 3D View ---
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1.set_title('3D Trajectory View (Colored by Speed)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y); ax1.set_zlim(0, SPACE_Z)
        ax1.set_box_aspect([1, 1, 1])
        
        # Color by speed
        max_speed = max(self.speeds) if self.speeds and max(self.speeds) > 0 else 1
        colors = plt.cm.plasma(np.array(self.speeds) / max_speed)
        
        for i in range(len(self.uav_positions) - 1):
            ax1.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    [self.uav_positions[i][2], self.uav_positions[i+1][2]],
                    color=colors[i], linewidth=2)
        
        # Plot obstacles as cylinders
        obstacle_colors_3d = plt.cm.Reds(np.linspace(0.4, 0.9, len(self.obstacles)))
        for idx, (obs, height) in enumerate(zip(self.obstacles, self.obstacle_heights)):
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, height, 10)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = obs[0] + self.env.obstacle_radius * np.cos(theta_grid)
            y_cyl = obs[1] + self.env.obstacle_radius * np.sin(theta_grid)
            ax1.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color=obstacle_colors_3d[idx])
        
        ax1.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                   c='green', marker='s', s=200, label='Base')
        ax1.legend(fontsize=8)
        
        # --- Top View with Obstacles ---
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.set_title('Top View (X-Y Plane) - 5 Obstacles', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.set_xlim(0, SPACE_X); ax2.set_ylim(0, SPACE_Y)
        ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
        
        for i in range(len(self.uav_positions) - 1):
            ax2.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    color=colors[i], linewidth=2)
        
        # Plot obstacles with height-based colors
        for idx, (obs, height) in enumerate(zip(self.obstacles, self.obstacle_heights)):
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color=obstacle_colors_3d[idx], alpha=0.4)
            ax2.add_patch(circle)
            ax2.text(obs[0], obs[1], f'{height}m', fontsize=9, ha='center', 
                    va='center', fontweight='bold', color='darkred')
        
        ax2.plot(self.base_position[0], self.base_position[1], 'gs', markersize=12, label='Base')
        ax2.legend(fontsize=8)
        
        # --- User Movement Path ---
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.set_title('User Movement Path', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
        ax3.set_xlim(0, SPACE_X); ax3.set_ylim(0, SPACE_Y)
        ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
        
        user_x = [pos[0] for pos in self.user_positions]
        user_y = [pos[1] for pos in self.user_positions]
        ax3.plot(user_x, user_y, 'r--', linewidth=2, alpha=0.6, label='User Path')
        ax3.plot(user_x[0], user_y[0], 'go', markersize=10, label='Start')
        ax3.plot(user_x[-1], user_y[-1], 'ro', markersize=10, label='End')
        
        # Highlight movement points (every 10 steps)
        for i in range(0, len(user_x), 10):
            ax3.plot(user_x[i], user_y[i], 'r.', markersize=5)
        
        ax3.legend(fontsize=8)
        
        # --- Data Rate & Altitude ---
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.set_title('Data Rate & Altitude', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Step'); ax4.set_ylabel('Data Rate (bps/Hz)', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue'); ax4.grid(True, alpha=0.3)
        
        steps = list(range(len(self.data_rates)))
        ax4.plot(steps, self.data_rates, 'b-', linewidth=2, label='Data Rate')
        
        ax4_alt = ax4.twinx()
        ax4_alt.set_ylabel('Altitude (m)', color='orange')
        ax4_alt.tick_params(axis='y', labelcolor='orange')
        ax4_alt.plot(steps, self.altitudes, 'orange', linewidth=2, label='Altitude')
        
        # Draw obstacle height lines
        for idx, height in enumerate(self.obstacle_heights):
            ax4_alt.axhline(y=height, color=obstacle_colors_3d[idx], 
                           linestyle='--', alpha=0.4, linewidth=1)
        
        # --- Speed ---
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.set_title('UAV Speed Over Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time Step'); ax5.set_ylabel('Speed (m/s)', color='green')
        ax5.tick_params(axis='y', labelcolor='green'); ax5.grid(True, alpha=0.3)
        ax5.plot(steps, self.speeds, 'g-', linewidth=2, label='Speed')
        avg_speed_val = np.mean(self.speeds)
        ax5.axhline(y=avg_speed_val, color='red', linestyle='--', alpha=0.5, 
                    label=f'Avg: {avg_speed_val:.2f} m/s')
        ax5.legend(loc='upper right', fontsize=8)
        
        # --- Cumulative Distance ---
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.set_title('Cumulative Distance Traveled', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time Step'); ax6.set_ylabel('Distance (m)', color='purple')
        ax6.tick_params(axis='y', labelcolor='purple'); ax6.grid(True, alpha=0.3)
        ax6.plot(steps, self.cumulative_distances, 'purple', linewidth=2.5, label='Total Distance')
        final_dist = self.cumulative_distances[-1] if self.cumulative_distances else 0
        ax6.text(0.98, 0.02, f'Final: {final_dist:.2f} m', transform=ax6.transAxes,
                fontsize=11, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax6.legend(loc='upper left', fontsize=8)
        
        # --- Velocity Vector Components ---
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.set_title('Velocity Components (vx, vy, vz)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Time Step'); ax7.set_ylabel('Velocity (m/s)')
        ax7.grid(True, alpha=0.3)
        
        vx = [v[0] for v in self.velocities]
        vy = [v[1] for v in self.velocities]
        vz = [v[2] for v in self.velocities]
        
        ax7.plot(steps, vx, 'r-', linewidth=1.5, alpha=0.7, label='vx')
        ax7.plot(steps, vy, 'g-', linewidth=1.5, alpha=0.7, label='vy')
        ax7.plot(steps, vz, 'b-', linewidth=1.5, alpha=0.7, label='vz')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax7.legend(loc='upper right', fontsize=8)
        
        # --- Step Distance ---
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.set_title('Distance per Step', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Time Step'); ax8.set_ylabel('Distance (m)', color='brown')
        ax8.tick_params(axis='y', labelcolor='brown'); ax8.grid(True, alpha=0.3)
        ax8.plot(steps, self.step_distances, 'brown', linewidth=1.5, alpha=0.7)
        ax8.fill_between(steps, self.step_distances, alpha=0.3, color='brown')
        avg_step_dist = np.mean(self.step_distances)
        ax8.axhline(y=avg_step_dist, color='red', linestyle='--', alpha=0.5,
                   label=f'Avg: {avg_step_dist:.2f} m')
        ax8.legend(loc='upper right', fontsize=8)
        
        # --- Statistics Panel ---
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis('off')
        ax9.set_title('Episode Statistics', fontsize=12, fontweight='bold')
        
        # Calculate user movement
        user_distances = []
        for i in range(len(self.user_positions) - 1):
            dx = self.user_positions[i+1][0] - self.user_positions[i][0]
            dy = self.user_positions[i+1][1] - self.user_positions[i][1]
            user_distances.append(np.sqrt(dx**2 + dy**2))
        total_user_movement = sum(user_distances)
        
        stats_text = (
            f'╔════════════════════════════════════╗\n'
            f'║   TRAJECTORY STATISTICS            ║\n'
            f'╚════════════════════════════════════╝\n\n'
            f'🏢 ENVIRONMENT:\n'
            f'  • Obstacles:       {len(self.obstacles)}\n'
            f'  • Heights Range:   {min(self.obstacle_heights)}-{max(self.obstacle_heights)} m\n'
            f'  • User Movement:   Active\n\n'
            f'✈️  UAV METRICS:\n'
            f'  • Total Distance:  {self.cumulative_distances[-1]:8.2f} m\n'
            f'  • Avg Speed:       {np.mean(self.speeds):8.2f} m/s\n'
            f'  • Max Speed:       {np.max(self.speeds):8.2f} m/s\n'
            f'  • Min Speed:       {np.min(self.speeds):8.2f} m/s\n\n'
            f'👤 USER METRICS:\n'
            f'  • Total Movement:  {total_user_movement:8.2f} m\n'
            f'  • Move Steps:      {len([d for d in user_distances if d > 0]):8d}\n\n'
            f'📡 DATA RATE:\n'
            f'  • Avg Data Rate:   {np.mean(self.data_rates):8.2f} bps/Hz\n'
            f'  • Max Data Rate:   {np.max(self.data_rates):8.2f} bps/Hz\n'
            f'  • Min Data Rate:   {np.min(self.data_rates):8.2f} bps/Hz\n\n'
            f'📏 ALTITUDE:\n'
            f'  • Avg Altitude:    {np.mean(self.altitudes):8.1f} m\n'
            f'  • Max Altitude:    {np.max(self.altitudes):8.1f} m\n'
            f'  • Min Altitude:    {np.min(self.altitudes):8.1f} m\n\n'
            f'🎯 REWARDS:\n'
            f'  • Total Reward:    {np.sum(self.rewards):8.1f}\n'
            f'  • Avg Reward:      {np.mean(self.rewards):8.2f}\n'
        )
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Static plot saved to: {save_path}")
        plt.close()
    
    def visualize_all(self, output_dir="visualizations"):
        """Generate all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n╔{'═'*68}╗")
        print(f"║  UAV TRAJECTORY VISUALIZATION - 5 OBSTACLES & USER MOVEMENT     ║")
        print(f"╚{'═'*68}╝\n")
        
        for episode in range(self.num_episodes):
            self.collect_trajectory(episode)
            episode_suffix = f"_ep{episode+1}" if self.num_episodes > 1 else ""
            
            self.create_static_plot(save_path=os.path.join(output_dir, f"trajectory_static{episode_suffix}.png"))
            self.create_2d_animation(save_path=os.path.join(output_dir, f"trajectory_2d{episode_suffix}.gif"))
            self.create_3d_animation(save_path=os.path.join(output_dir, f"trajectory_3d{episode_suffix}.gif"))
        
        print(f"\n{'='*60}")
        print(f"✓ All visualizations saved to '{output_dir}/' directory")
        print(f"{'='*60}\n")
        self.env.close()

if __name__ == "__main__":
    # Update this path to your trained model
    MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\fyp RL\models\uav_ppo\1769524969\final_model_continuous.zip"
    
    if os.path.exists(MODEL_PATH):
        visualizer = UAVTrajectoryVisualizer(model_path=MODEL_PATH, num_episodes=1, max_steps=200)
        visualizer.visualize_all(output_dir="visualizations")
        print("Done! Check the 'visualizations' folder for outputs.")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"Please update MODEL_PATH in the script to point to your trained model.")
