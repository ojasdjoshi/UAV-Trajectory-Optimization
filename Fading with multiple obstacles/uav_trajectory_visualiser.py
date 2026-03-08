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
    Shows obstacles, user movement, performance metrics, velocity, and distance.
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
        # UPDATED: Store (x, y, height) for each obstacle from the environment
        self.obstacles = [(o.x, o.y, o.z) for o in self.env.obstacles]
        
        print(f"\n{'='*70}")
        print(f"  EPISODE {episode_num + 1}: TRAJECTORY COLLECTION")
        print(f"{'='*70}")
        
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
        
        print(f"\n{'─'*70}")
        print(f"  EPISODE {episode_num + 1} SUMMARY")
        print(f"{'─'*70}")
        print(f"  Total Steps:          {step}")
        print(f"  Total Reward:         {total_reward:.2f}")
        print(f"\n  DISTANCE & SPEED METRICS:")
        print(f"  ├─ Total Distance:    {total_distance:.2f} m")
        print(f"  ├─ Average Speed:     {avg_speed:.2f} m/s")
        print(f"  ├─ Maximum Speed:     {max_speed:.2f} m/s")
        print(f"  └─ Minimum Speed:     {min_speed:.2f} m/s")
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
        Create animated 2D top-down view of the trajectory.
        """
        print(f"  [→] Creating 2D animation...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- Top Left: Top-down view ---
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_xlabel('X Position (m)', fontsize=11)
        ax1.set_ylabel('Y Position (m)', fontsize=11)
        ax1.set_title('UAV Trajectory - Top View', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot static elements
        ax1.plot(self.base_position[0], self.base_position[1], 
                'gs', markersize=15, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        # UPDATED: Visualize obstacles as circles (top-view)
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color='red', alpha=0.3, label='Obstacle' if obs == self.obstacles[0] else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'rx', markersize=10, markeredgewidth=2)
        
        # Initialize dynamic elements
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        user_point, = ax1.plot([], [], 'mo', markersize=10, label='User', markeredgecolor='purple', markeredgewidth=2)
        
        ax1.legend(loc='upper right', fontsize=9)
        
        # --- Top Right: Data Rate & Altitude ---
        ax2.set_xlim(0, len(self.uav_positions))
        ax2.set_ylim(0, max(max(self.data_rates) * 1.1, 10))
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Data Rate (bps/Hz)', fontsize=11, color='blue')
        ax2.set_title('Data Rate & Altitude', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        data_rate_line, = ax2.plot([], [], 'b-', linewidth=2, label='Data Rate')
        
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylim(0, SPACE_Z)
        ax2_alt.set_ylabel('Altitude (m)', fontsize=11, color='orange')
        ax2_alt.tick_params(axis='y', labelcolor='orange')
        altitude_line, = ax2_alt.plot([], [], 'orange', linewidth=2, label='Altitude')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_alt.get_legend_handles_labels()
        ax2.legend(lines1 + labels2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # --- Bottom Left: Speed ---
        ax3.set_xlim(0, len(self.uav_positions))
        ax3.set_ylim(0, max(max(self.speeds) * 1.1, 1))
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Speed (m/s)', fontsize=11, color='green')
        ax3.set_title('UAV Speed', fontsize=13, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.grid(True, alpha=0.3)
        
        speed_line, = ax3.plot([], [], 'g-', linewidth=2, label='Speed')
        avg_speed = np.mean(self.speeds)
        ax3.axhline(y=avg_speed, color='red', linestyle='--', linewidth=1.5, alpha=0.6, 
                    label=f'Avg: {avg_speed:.2f} m/s')
        ax3.legend(loc='upper left', fontsize=9)
        
        # --- Bottom Right: Cumulative Distance ---
        ax4.set_xlim(0, len(self.uav_positions))
        ax4.set_ylim(0, max(self.cumulative_distances) * 1.1 if self.cumulative_distances else 1)
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Distance (m)', fontsize=11, color='purple')
        ax4.set_title('Cumulative Distance Traveled', fontsize=13, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.grid(True, alpha=0.3)
        
        distance_line, = ax4.plot([], [], 'purple', linewidth=2.5, label='Total Distance')
        total_dist = self.cumulative_distances[-1] if self.cumulative_distances else 0
        ax4.text(0.98, 0.02, f'Final: {total_dist:.2f} m', transform=ax4.transAxes,
                fontsize=10, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax4.legend(loc='upper left', fontsize=9)
        
        # Text annotations
        step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                            fontsize=10, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        def init():
            uav_trail.set_data([], [])
            uav_point.set_data([], [])
            user_point.set_data([], [])
            data_rate_line.set_data([], [])
            altitude_line.set_data([], [])
            speed_line.set_data([], [])
            distance_line.set_data([], [])
            step_text.set_text('')
            return (uav_trail, uav_point, user_point, data_rate_line, 
                    altitude_line, speed_line, distance_line, step_text)
        
        def animate(frame):
            # Update UAV trail and position
            x_trail = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_trail = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_trail, y_trail)
            uav_point.set_data([self.uav_positions[frame][0]], [self.uav_positions[frame][1]])
            
            # Update user position
            user_point.set_data([self.user_positions[frame][0]], [self.user_positions[frame][1]])
            
            # Update metrics
            steps = list(range(frame + 1))
            data_rate_line.set_data(steps, self.data_rates[:frame+1])
            altitude_line.set_data(steps, self.altitudes[:frame+1])
            speed_line.set_data(steps, self.speeds[:frame+1])
            distance_line.set_data(steps, self.cumulative_distances[:frame+1])
            
            # Update text with speed and distance highlighted
            step_text.set_text(
                f'Step:     {frame + 1:3d}/{len(self.uav_positions)}\n'
                f'Speed:    {self.speeds[frame]:6.2f} m/s\n'
                f'Step Dst: {self.step_distances[frame]:6.2f} m\n'
                f'Total Dst:{self.cumulative_distances[frame]:7.2f} m\n'
                f'Data Rate:{self.data_rates[frame]:6.2f} bps/Hz\n'
                f'Altitude: {self.altitudes[frame]:6.1f} m'
            )
            
            return (uav_trail, uav_point, user_point, data_rate_line, 
                    altitude_line, speed_line, distance_line, step_text)
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=True, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"      ✓ 2D animation saved to: {save_path}")
        
        plt.close()
    
    def create_3d_animation(self, save_path="uav_trajectory_3d.gif", interval=100):
        """
        Create animated 3D view of the trajectory with velocity vectors.
        """
        print(f"  [→] Creating 3D animation...")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, SPACE_X)
        ax.set_ylim(0, SPACE_Y)
        ax.set_zlim(0, SPACE_Z)
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title('UAV 3D Trajectory with Velocity Vectors', fontsize=14, fontweight='bold', pad=20)
        
        # Plot static elements
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                  c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        # UPDATED: Render each obstacle using its specific height (obs[2])
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(0, obs[2], 2) # Use obs[2] as the height instead of SPACE_Z
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='red')
        
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7, label='UAV Path')
        uav_point = ax.scatter([], [], [], c='blue', marker='o', s=150, edgecolors='darkblue', linewidths=2)
        user_point = ax.scatter([], [], [], c='magenta', marker='o', s=100, edgecolors='purple', linewidths=2, label='User')
        
        # Velocity vector (quiver)
        velocity_arrow = None
        
        step_text = fig.text(0.02, 0.95, '', fontsize=10, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax.legend(loc='upper right', fontsize=10)
        
        def init():
            uav_trail.set_data([], [])
            uav_trail.set_3d_properties([])
            step_text.set_text('')
            return uav_trail, step_text
        
        def animate(frame):
            nonlocal velocity_arrow
            
            x_trail = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_trail = [pos[1] for pos in self.uav_positions[:frame+1]]
            z_trail = [pos[2] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_trail, y_trail)
            uav_trail.set_3d_properties(z_trail)
            
            uav_point._offsets3d = ([self.uav_positions[frame][0]], 
                                    [self.uav_positions[frame][1]], 
                                    [self.uav_positions[frame][2]])
            
            user_point._offsets3d = ([self.user_positions[frame][0]], 
                                     [self.user_positions[frame][1]], 
                                     [self.user_positions[frame][2]])
            
            # Draw velocity vector
            if velocity_arrow:
                velocity_arrow.remove()
            
            vel = self.velocities[frame]
            vel_scale = 5  # Scale for visibility
            velocity_arrow = ax.quiver(
                self.uav_positions[frame][0], 
                self.uav_positions[frame][1], 
                self.uav_positions[frame][2],
                vel[0] * vel_scale, vel[1] * vel_scale, vel[2] * vel_scale,
                color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.7
            )
            
            step_text.set_text(
                f'Step:      {frame + 1:3d}/{len(self.uav_positions)}\n'
                f'Speed:     {self.speeds[frame]:6.2f} m/s\n'
                f'Distance:  {self.cumulative_distances[frame]:7.2f} m\n'
                f'Data Rate: {self.data_rates[frame]:6.2f} bps/Hz\n'
                f'Altitude:  {self.altitudes[frame]:6.1f} m'
            )
            
            ax.view_init(elev=20, azim=45 + frame * 0.5)
            
            return uav_trail, uav_point, user_point, step_text
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=False, repeat=True)
        
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"      ✓ 3D animation saved to: {save_path}")
        
        plt.close()
    
    def create_static_plot(self, save_path="uav_trajectory_static.png"):
        """
        Create a comprehensive static plot with multiple views including velocity and distance.
        """
        print(f"  [→] Creating static trajectory plot...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # --- 3D View ---
        ax1 = fig.add_subplot(3, 2, 1, projection='3d')
        ax1.set_title('3D Trajectory View (Colored by Speed)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y); ax1.set_zlim(0, SPACE_Z)
        ax1.set_box_aspect([1, 1, 1])
        
        # UPDATED: Draw 3D obstacles with correct heights in static plot
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, obs[2], 2)
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.1, color='red')

        # Color by speed
        max_speed = max(self.speeds) if self.speeds and max(self.speeds) > 0 else 1
        colors = plt.cm.plasma(np.array(self.speeds) / max_speed)
        
        for i in range(len(self.uav_positions) - 1):
            ax1.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    [self.uav_positions[i][2], self.uav_positions[i+1][2]],
                    color=colors[i], linewidth=2)
        
        ax1.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                   c='green', marker='s', s=200, label='Base')
        ax1.legend()
        
        # --- Top View ---
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.set_title('Top View (X-Y Plane) - Colored by Speed', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.set_xlim(0, SPACE_X); ax2.set_ylim(0, SPACE_Y)
        ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
        
        for i in range(len(self.uav_positions) - 1):
            ax2.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    color=colors[i], linewidth=2)
        
        # UPDATED: Ensure top view shows obstacles
        for obs in self.obstacles:
            ax2.add_patch(Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.3))
        
        # --- Data Rate & Altitude ---
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.set_title('Data Rate & Altitude', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Step'); ax3.set_ylabel('Data Rate (bps/Hz)', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue'); ax3.grid(True, alpha=0.3)
        
        steps = list(range(len(self.data_rates)))
        ax3.plot(steps, self.data_rates, 'b-', linewidth=2, label='Data Rate')
        
        ax3_alt = ax3.twinx()
        ax3_alt.set_ylabel('Altitude (m)', color='orange')
        ax3_alt.tick_params(axis='y', labelcolor='orange')
        ax3_alt.plot(steps, self.altitudes, 'orange', linewidth=2, label='Altitude')
        
        # --- Speed ---
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.set_title('UAV Speed Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Step'); ax4.set_ylabel('Speed (m/s)', color='green')
        ax4.tick_params(axis='y', labelcolor='green'); ax4.grid(True, alpha=0.3)
        ax4.plot(steps, self.speeds, 'g-', linewidth=2, label='Speed')
        avg_speed_val = np.mean(self.speeds)
        ax4.axhline(y=avg_speed_val, color='red', linestyle='--', alpha=0.5, 
                    label=f'Avg: {avg_speed_val:.2f} m/s')
        ax4.legend(loc='upper right')
        
        # --- Cumulative Distance ---
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.set_title('Cumulative Distance Traveled', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time Step'); ax5.set_ylabel('Distance (m)', color='purple')
        ax5.tick_params(axis='y', labelcolor='purple'); ax5.grid(True, alpha=0.3)
        ax5.plot(steps, self.cumulative_distances, 'purple', linewidth=2.5, label='Total Distance')
        final_dist = self.cumulative_distances[-1] if self.cumulative_distances else 0
        ax5.text(0.98, 0.02, f'Final: {final_dist:.2f} m', transform=ax5.transAxes,
                fontsize=11, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax5.legend(loc='upper left')
        
        # --- Statistics Panel ---
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')
        ax6.set_title('Episode Statistics', fontsize=12, fontweight='bold')
        
        stats_text = (
            f'╔══════════════════════════════════════╗\n'
            f'║   TRAJECTORY STATISTICS              ║\n'
            f'╚══════════════════════════════════════╝\n\n'
            f'📏 DISTANCE & SPEED:\n'
            f'  • Total Distance:  {self.cumulative_distances[-1]:8.2f} m\n'
            f'  • Avg Speed:       {np.mean(self.speeds):8.2f} m/s\n'
            f'  • Max Speed:       {np.max(self.speeds):8.2f} m/s\n'
            f'  • Min Speed:       {np.min(self.speeds):8.2f} m/s\n\n'
            f'📡 DATA RATE:\n'
            f'  • Avg Data Rate:   {np.mean(self.data_rates):8.2f} bps/Hz\n'
            f'  • Max Data Rate:   {np.max(self.data_rates):8.2f} bps/Hz\n'
            f'  • Min Data Rate:   {np.min(self.data_rates):8.2f} bps/Hz\n\n'
            f'✈️  ALTITUDE:\n'
            f'  • Avg Altitude:    {np.mean(self.altitudes):8.1f} m\n'
            f'  • Max Altitude:    {np.max(self.altitudes):8.1f} m\n'
            f'  • Min Altitude:    {np.min(self.altitudes):8.1f} m\n\n'
            f'🎯 REWARDS:\n'
            f'  • Total Reward:    {np.sum(self.rewards):8.1f}\n'
            f'  • Avg Reward:      {np.mean(self.rewards):8.2f}\n'
        )
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=9.5, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Static plot saved to: {save_path}")
        plt.close()
    
    def visualize_all(self, output_dir="visualizations"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n╔{'═'*68}╗")
        print(f"║  UAV TRAJECTORY VISUALIZATION WITH VELOCITY & DISTANCE          ║")
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
    # Ensure this path matches your trained model
    MODEL_PATH = r"models\uav_ppo\1770228867\final_model_continuous.zip"
    
    if os.path.exists(MODEL_PATH):
        visualizer = UAVTrajectoryVisualizer(model_path=MODEL_PATH, num_episodes=1, max_steps=200)
        visualizer.visualize_all(output_dir="visualizations")
        print("Done! Check the 'visualizations' folder for outputs.")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        
        