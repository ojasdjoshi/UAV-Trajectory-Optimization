import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import PPO
from uavenv import UavEnv, SPACE_X, SPACE_Y, SPACE_Z, MIN_ALTITUDE, MAX_ALTITUDE
import os

class UAVTrajectoryVisualizer:
    """
    Enhanced visualizer for UAV trajectories with multi-episode support.
    Features:
    - Multiple episode visualization and comparison
    - Varied obstacle heights for better realism
    - Aggregate statistics across episodes
    - Side-by-side episode comparisons
    """
    
    def __init__(self, model_path, num_episodes=3, max_steps=200):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained PPO model
            num_episodes: Number of episodes to visualize (default: 3)
            max_steps: Maximum steps per episode
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Initialize environment
        self.env = UavEnv()
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, env=self.env)
        
        # Storage for ALL episodes data
        self.all_episodes_data = []
        
        # Storage for current episode
        self.reset_trajectory_data()
        
    def reset_trajectory_data(self):
        """Reset trajectory tracking for current episode."""
        self.uav_positions = []
        self.user_positions = []
        self.base_position = None
        self.obstacles = []
        self.obstacle_heights = []  # NEW: Store individual obstacle heights
        self.data_rates = []
        self.rewards = []
        self.altitudes = []
        self.speeds = []
        self.velocities = []
        self.step_distances = []
        self.cumulative_distances = []
        
    def randomize_obstacle_heights(self):
        """
        Modify obstacle heights to have varied heights instead of uniform.
        Heights will range from 20m to 80m for better visualization.
        """
        self.obstacle_heights = []
        for i, obs in enumerate(self.env.obstacles):
            # Generate random height between 20-80m (or use bounds from environment)
            min_height = max(20, MIN_ALTITUDE)
            max_height = min(80, MAX_ALTITUDE)
            random_height = np.random.uniform(min_height, max_height)
            
            # Update the obstacle's z position
            obs.z = random_height
            self.obstacle_heights.append(random_height)
            
        print(f"  [✓] Obstacle heights randomized: {[f'{h:.1f}m' for h in self.obstacle_heights]}")
    
    def collect_trajectory(self, episode_num=0):
        """
        Run one episode and collect trajectory data.
        
        Args:
            episode_num: Episode number for display
        """
        self.reset_trajectory_data()
        
        obs, info = self.env.reset()
        
        # Randomize obstacle heights for this episode
        self.randomize_obstacle_heights()
        
        done = False
        step = 0
        total_reward = 0
        
        # Store initial positions
        self.base_position = (self.env.base.x, self.env.base.y, self.env.base.z)
        self.obstacles = [(o.x, o.y, o.z) for o in self.env.obstacles]
        
        print(f"\n{'='*70}")
        print(f"  EPISODE {episode_num + 1}/{self.num_episodes}: TRAJECTORY COLLECTION")
        print(f"{'='*70}")
        
        while not done and step < self.max_steps:
            # Predict action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            self.uav_positions.append((self.env.uav.x, self.env.uav.y, self.env.uav.z))
            self.user_positions.append((self.env.user.x, self.env.user.y, self.env.user.z))
            self.data_rates.append(info.get('data_rate', 0))
            self.rewards.append(reward)
            self.altitudes.append(self.env.uav.z)
            self.speeds.append(info.get('uav_speed', 0))
            self.velocities.append(info.get('uav_velocity', np.array([0, 0, 0])))
            self.step_distances.append(info.get('step_distance', 0))
            self.cumulative_distances.append(info.get('total_distance', 0))
            
            total_reward += reward
            step += 1
            
            # Display progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step:3d}/{self.max_steps} | "
                      f"Speed: {self.speeds[-1]:5.2f} m/s | "
                      f"Total Dist: {self.cumulative_distances[-1]:7.2f} m | "
                      f"Data Rate: {self.data_rates[-1]:5.2f} bps/Hz")
        
        # Calculate episode statistics
        episode_stats = {
            'episode_num': episode_num + 1,
            'total_steps': step,
            'total_reward': total_reward,
            'avg_speed': np.mean(self.speeds),
            'max_speed': np.max(self.speeds),
            'min_speed': np.min(self.speeds),
            'total_distance': self.cumulative_distances[-1] if self.cumulative_distances else 0,
            'avg_data_rate': np.mean(self.data_rates),
            'max_data_rate': np.max(self.data_rates),
            'min_data_rate': np.min(self.data_rates),
            'avg_altitude': np.mean(self.altitudes),
            'max_altitude': np.max(self.altitudes),
            'min_altitude': np.min(self.altitudes),
            'crashed': terminated and step < self.max_steps
        }
        
        # Print episode summary
        print(f"\n{'─'*70}")
        print(f"  EPISODE {episode_num + 1} SUMMARY")
        print(f"{'─'*70}")
        print(f"  Total Steps:          {step}")
        print(f"  Total Reward:         {total_reward:.2f}")
        print(f"\n  DISTANCE & SPEED:")
        print(f"  ├─ Total Distance:    {episode_stats['total_distance']:.2f} m")
        print(f"  ├─ Average Speed:     {episode_stats['avg_speed']:.2f} m/s")
        print(f"  └─ Max Speed:         {episode_stats['max_speed']:.2f} m/s")
        print(f"\n  DATA RATE:")
        print(f"  ├─ Average:           {episode_stats['avg_data_rate']:.2f} bps/Hz")
        print(f"  └─ Max:               {episode_stats['max_data_rate']:.2f} bps/Hz")
        print(f"\n  ALTITUDE:")
        print(f"  ├─ Average:           {episode_stats['avg_altitude']:.1f} m")
        print(f"  └─ Max:               {episode_stats['max_altitude']:.1f} m")
        print(f"\n  STATUS:")
        print(f"  └─ Crashed:           {'YES ❌' if episode_stats['crashed'] else 'NO ✓'}")
        print(f"{'='*70}\n")
        
        # Store episode data
        episode_data = {
            'uav_positions': self.uav_positions.copy(),
            'user_positions': self.user_positions.copy(),
            'base_position': self.base_position,
            'obstacles': self.obstacles.copy(),
            'obstacle_heights': self.obstacle_heights.copy(),
            'data_rates': self.data_rates.copy(),
            'rewards': self.rewards.copy(),
            'altitudes': self.altitudes.copy(),
            'speeds': self.speeds.copy(),
            'cumulative_distances': self.cumulative_distances.copy(),
            'stats': episode_stats
        }
        
        self.all_episodes_data.append(episode_data)
        
        return step
    
    def create_multi_episode_comparison(self, save_path="multi_episode_comparison.png"):
        """
        Create a comprehensive comparison plot showing all episodes side by side.
        """
        print(f"  [→] Creating multi-episode comparison plot...")
        
        n_episodes = len(self.all_episodes_data)
        colors = plt.cm.tab10(np.linspace(0, 1, n_episodes))
        
        fig = plt.figure(figsize=(22, 14))
        
        # --- 3D View with all episodes ---
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.set_title('All Episodes - 3D Trajectories', fontsize=13, fontweight='bold')
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_zlim(0, SPACE_Z)
        
        # Plot base station (same for all episodes)
        base_pos = self.all_episodes_data[0]['base_position']
        ax1.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], 
                   c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        # Plot obstacles with varied heights
        obstacles = self.all_episodes_data[0]['obstacles']
        heights = self.all_episodes_data[0]['obstacle_heights']
        
        for i, (obs, height) in enumerate(zip(obstacles, heights)):
            # Draw cylinder for obstacle
            theta = np.linspace(0, 2*np.pi, 20)
            radius = self.env.obstacle_radius
            x_circle = obs[0] + radius * np.cos(theta)
            y_circle = obs[1] + radius * np.sin(theta)
            z_bottom = np.zeros_like(theta)
            z_top = np.full_like(theta, height)
            
            # Draw sides
            for j in range(len(theta)-1):
                xs = [x_circle[j], x_circle[j+1], x_circle[j+1], x_circle[j]]
                ys = [y_circle[j], y_circle[j+1], y_circle[j+1], y_circle[j]]
                zs = [0, 0, height, height]
                verts = [list(zip(xs, ys, zs))]
                ax1.add_collection3d(Poly3DCollection(verts, alpha=0.3, facecolor='red', edgecolor='darkred'))
            
            # Top cap
            verts_top = [list(zip(x_circle, y_circle, z_top))]
            ax1.add_collection3d(Poly3DCollection(verts_top, alpha=0.4, facecolor='red'))
            
            # Add height label
            ax1.text(obs[0], obs[1], height + 5, f'{height:.0f}m', 
                    fontsize=8, ha='center', color='darkred', fontweight='bold')
        
        # Plot each episode trajectory
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            positions = ep_data['uav_positions']
            if len(positions) > 0:
                x = [p[0] for p in positions]
                y = [p[1] for p in positions]
                z = [p[2] for p in positions]
                ax1.plot(x, y, z, color=colors[ep_idx], linewidth=2.5, 
                        label=f'Episode {ep_idx+1}', alpha=0.8)
                # Mark start and end
                ax1.scatter([x[0]], [y[0]], [z[0]], color=colors[ep_idx], 
                           marker='o', s=100, edgecolors='black', linewidths=2)
                ax1.scatter([x[-1]], [y[-1]], [z[-1]], color=colors[ep_idx], 
                           marker='X', s=150, edgecolors='black', linewidths=2)
        
        ax1.legend(loc='upper right', fontsize=9)
        ax1.view_init(elev=25, azim=45)
        
        # --- Top View (X-Y) ---
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('All Episodes - Top View', fontsize=13, fontweight='bold')
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_xlim(0, SPACE_X)
        ax2.set_ylim(0, SPACE_Y)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot obstacles
        for obs in obstacles:
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax2.add_patch(circle)
        
        # Plot base
        ax2.plot(base_pos[0], base_pos[1], 'gs', markersize=15, 
                markeredgecolor='black', markeredgewidth=2, label='Base')
        
        # Plot trajectories
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            positions = ep_data['uav_positions']
            if len(positions) > 0:
                x = [p[0] for p in positions]
                y = [p[1] for p in positions]
                ax2.plot(x, y, color=colors[ep_idx], linewidth=2, 
                        label=f'Episode {ep_idx+1}', alpha=0.7)
                ax2.plot(x[0], y[0], 'o', color=colors[ep_idx], 
                        markersize=8, markeredgecolor='black', markeredgewidth=1.5)
                ax2.plot(x[-1], y[-1], 'X', color=colors[ep_idx], 
                        markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        
        ax2.legend(loc='upper right', fontsize=8)
        
        # --- Speed Comparison ---
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title('Speed Comparison Across Episodes', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Speed (m/s)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            speeds = ep_data['speeds']
            steps = range(len(speeds))
            ax3.plot(steps, speeds, color=colors[ep_idx], linewidth=2, 
                    label=f'Ep {ep_idx+1} (Avg: {ep_data["stats"]["avg_speed"]:.1f} m/s)', 
                    alpha=0.7)
        
        ax3.legend(loc='upper right', fontsize=8)
        
        # --- Data Rate Comparison ---
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Data Rate Comparison', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Time Step', fontsize=10)
        ax4.set_ylabel('Data Rate (bps/Hz)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            data_rates = ep_data['data_rates']
            steps = range(len(data_rates))
            ax4.plot(steps, data_rates, color=colors[ep_idx], linewidth=2, 
                    label=f'Episode {ep_idx+1}', alpha=0.7)
        
        ax4.legend(loc='upper right', fontsize=8)
        
        # --- Cumulative Distance Comparison ---
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title('Cumulative Distance Traveled', fontsize=13, fontweight='bold')
        ax5.set_xlabel('Time Step', fontsize=10)
        ax5.set_ylabel('Distance (m)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            distances = ep_data['cumulative_distances']
            steps = range(len(distances))
            final_dist = distances[-1] if distances else 0
            ax5.plot(steps, distances, color=colors[ep_idx], linewidth=2.5, 
                    label=f'Ep {ep_idx+1} (Total: {final_dist:.1f} m)', alpha=0.7)
        
        ax5.legend(loc='upper left', fontsize=8)
        
        # --- Statistics Summary ---
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        ax6.set_title('Aggregate Statistics', fontsize=13, fontweight='bold')
        
        # Calculate aggregate stats
        all_rewards = [ep['stats']['total_reward'] for ep in self.all_episodes_data]
        all_distances = [ep['stats']['total_distance'] for ep in self.all_episodes_data]
        all_avg_speeds = [ep['stats']['avg_speed'] for ep in self.all_episodes_data]
        all_avg_data_rates = [ep['stats']['avg_data_rate'] for ep in self.all_episodes_data]
        crashes = sum([1 for ep in self.all_episodes_data if ep['stats']['crashed']])
        
        stats_text = (
            f'╔══════════════════════════════════════╗\n'
            f'║  MULTI-EPISODE PERFORMANCE SUMMARY   ║\n'
            f'╚══════════════════════════════════════╝\n\n'
            f'📊 EPISODES: {n_episodes}\n\n'
            f'🏆 REWARDS:\n'
            f'  • Total:       {sum(all_rewards):10.1f}\n'
            f'  • Average:     {np.mean(all_rewards):10.1f}\n'
            f'  • Best:        {max(all_rewards):10.1f}\n'
            f'  • Worst:       {min(all_rewards):10.1f}\n\n'
            f'📏 DISTANCE:\n'
            f'  • Avg Total:   {np.mean(all_distances):10.1f} m\n'
            f'  • Max Total:   {max(all_distances):10.1f} m\n'
            f'  • Min Total:   {min(all_distances):10.1f} m\n\n'
            f'⚡ SPEED:\n'
            f'  • Avg (Mean):  {np.mean(all_avg_speeds):10.2f} m/s\n'
            f'  • Best Avg:    {max(all_avg_speeds):10.2f} m/s\n\n'
            f'📡 DATA RATE:\n'
            f'  • Avg (Mean):  {np.mean(all_avg_data_rates):10.2f} bps/Hz\n'
            f'  • Best Avg:    {max(all_avg_data_rates):10.2f} bps/Hz\n\n'
            f'💥 CRASHES:\n'
            f'  • Total:       {crashes}/{n_episodes}\n'
            f'  • Success Rate: {100*(n_episodes-crashes)/n_episodes:.1f}%\n'
        )
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Multi-episode comparison saved to: {save_path}")
        plt.close()
    
    def create_2d_animation(self, save_path, episode_num, interval=100):
        """
        Create animated 2D top-down view of the trajectory for a specific episode.
        """
        print(f"      [→] Creating 2D animation for Episode {episode_num}...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- Top Left: Top-down view ---
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_xlabel('X Position (m)', fontsize=11)
        ax1.set_ylabel('Y Position (m)', fontsize=11)
        ax1.set_title(f'Episode {episode_num} - UAV Trajectory (Top View)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot static elements
        ax1.plot(self.base_position[0], self.base_position[1], 
                'gs', markersize=15, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        for obs, height in zip(self.obstacles, self.obstacle_heights):
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color='red', alpha=0.3, label='Obstacle' if obs == self.obstacles[0] else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'rx', markersize=10, markeredgewidth=2)
            ax1.text(obs[0], obs[1], f'{height:.0f}m', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Initialize dynamic elements
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        user_point, = ax1.plot([], [], 'mo', markersize=10, label='User', markeredgecolor='darkmagenta', markeredgewidth=2)
        
        ax1.legend(loc='upper right', fontsize=10)
        
        # --- Top Right: Data Rate ---
        ax2.set_xlim(0, len(self.data_rates))
        ax2.set_ylim(0, max(self.data_rates) * 1.1 if self.data_rates else 10)
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Data Rate (bps/Hz)', fontsize=11)
        ax2.set_title('Data Rate Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        data_line, = ax2.plot([], [], 'b-', linewidth=2)
        data_point, = ax2.plot([], [], 'ro', markersize=8)
        
        # --- Bottom Left: Speed ---
        ax3.set_xlim(0, len(self.speeds))
        ax3.set_ylim(0, max(self.speeds) * 1.1 if self.speeds else 10)
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Speed (m/s)', fontsize=11)
        ax3.set_title('UAV Speed', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        speed_line, = ax3.plot([], [], 'g-', linewidth=2)
        speed_point, = ax3.plot([], [], 'ro', markersize=8)
        
        # --- Bottom Right: Altitude ---
        ax4.set_xlim(0, len(self.altitudes))
        ax4.set_ylim(0, SPACE_Z)
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Altitude (m)', fontsize=11)
        ax4.set_title('UAV Altitude', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add obstacle height reference lines
        for height in self.obstacle_heights:
            ax4.axhline(y=height, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        alt_line, = ax4.plot([], [], 'orange', linewidth=2)
        alt_point, = ax4.plot([], [], 'ro', markersize=8)
        
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            uav_trail.set_data([], [])
            uav_point.set_data([], [])
            user_point.set_data([], [])
            data_line.set_data([], [])
            data_point.set_data([], [])
            speed_line.set_data([], [])
            speed_point.set_data([], [])
            alt_line.set_data([], [])
            alt_point.set_data([], [])
            info_text.set_text('')
            return uav_trail, uav_point, user_point, data_line, data_point, speed_line, speed_point, alt_line, alt_point, info_text
        
        def animate(frame):
            # Update UAV trail and position
            x_data = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_data = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_data, y_data)
            uav_point.set_data([self.uav_positions[frame][0]], [self.uav_positions[frame][1]])
            
            # Update user position
            user_point.set_data([self.user_positions[frame][0]], [self.user_positions[frame][1]])
            
            # Update data rate
            steps = list(range(frame+1))
            data_line.set_data(steps, self.data_rates[:frame+1])
            data_point.set_data([frame], [self.data_rates[frame]])
            
            # Update speed
            speed_line.set_data(steps, self.speeds[:frame+1])
            speed_point.set_data([frame], [self.speeds[frame]])
            
            # Update altitude
            alt_line.set_data(steps, self.altitudes[:frame+1])
            alt_point.set_data([frame], [self.altitudes[frame]])
            
            # Update info text
            info_text.set_text(
                f'Episode {episode_num}\n'
                f'Step: {frame+1}/{len(self.uav_positions)}\n'
                f'Speed: {self.speeds[frame]:.2f} m/s\n'
                f'Distance: {self.cumulative_distances[frame]:.2f} m\n'
                f'Altitude: {self.altitudes[frame]:.1f} m'
            )
            
            return uav_trail, uav_point, user_point, data_line, data_point, speed_line, speed_point, alt_line, alt_point, info_text
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=False, repeat=True)
        
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"          ✓ 2D animation saved")
        
        plt.close()
    
    def create_3d_animation(self, save_path, episode_num, interval=100):
        """
        Create animated 3D view of the trajectory for a specific episode.
        """
        print(f"      [→] Creating 3D animation for Episode {episode_num}...")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, SPACE_X)
        ax.set_ylim(0, SPACE_Y)
        ax.set_zlim(0, SPACE_Z)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'Episode {episode_num} - 3D UAV Trajectory', fontsize=14, fontweight='bold')
        
        # Plot base station
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                  c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        # Plot obstacles as cylinders with varied heights
        for obs, height in zip(self.obstacles, self.obstacle_heights):
            theta = np.linspace(0, 2*np.pi, 20)
            radius = self.env.obstacle_radius
            x_circle = obs[0] + radius * np.cos(theta)
            y_circle = obs[1] + radius * np.sin(theta)
            
            # Draw cylinder
            for j in range(len(theta)-1):
                xs = [x_circle[j], x_circle[j+1], x_circle[j+1], x_circle[j]]
                ys = [y_circle[j], y_circle[j+1], y_circle[j+1], y_circle[j]]
                zs = [0, 0, height, height]
                verts = [list(zip(xs, ys, zs))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
                                                    facecolor='red', edgecolor='darkred'))
            
            # Top cap
            z_top = np.full_like(theta, height)
            verts_top = [list(zip(x_circle, y_circle, z_top))]
            ax.add_collection3d(Poly3DCollection(verts_top, alpha=0.4, facecolor='red'))
            
            # Height label
            ax.text(obs[0], obs[1], height + 5, f'{height:.0f}m', 
                   fontsize=9, ha='center', color='darkred', fontweight='bold')
        
        # Initialize dynamic elements
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, label='UAV Path')
        uav_point, = ax.plot([], [], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        user_point, = ax.plot([], [], [], 'mo', markersize=10, label='User', markeredgecolor='darkmagenta', markeredgewidth=2)
        
        ax.legend(loc='upper right', fontsize=10)
        
        step_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                             verticalalignment='top', family='monospace',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            uav_trail.set_data([], [])
            uav_trail.set_3d_properties([])
            uav_point.set_data([], [])
            uav_point.set_3d_properties([])
            user_point.set_data([], [])
            user_point.set_3d_properties([])
            step_text.set_text('')
            return uav_trail, uav_point, user_point, step_text
        
        def animate(frame):
            # Update UAV trail
            x_data = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_data = [pos[1] for pos in self.uav_positions[:frame+1]]
            z_data = [pos[2] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_data, y_data)
            uav_trail.set_3d_properties(z_data)
            
            # Update UAV point
            uav_point.set_data([self.uav_positions[frame][0]], [self.uav_positions[frame][1]])
            uav_point.set_3d_properties([self.uav_positions[frame][2]])
            
            # Update user point
            user_point.set_data([self.user_positions[frame][0]], [self.user_positions[frame][1]])
            user_point.set_3d_properties([self.user_positions[frame][2]])
            
            # Update text
            step_text.set_text(
                f'Episode:   {episode_num}\n'
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
        print(f"          ✓ 3D animation saved")
        
        plt.close()
    
    def create_individual_episode_plots(self, output_dir="visualizations"):
        """
        Create individual detailed plots and animations for each episode.
        """
        print(f"  [→] Creating individual episode visualizations...")
        
        for ep_idx, ep_data in enumerate(self.all_episodes_data):
            print(f"\n  Processing Episode {ep_idx+1}:")
            
            # Restore episode data
            self.uav_positions = ep_data['uav_positions']
            self.user_positions = ep_data['user_positions']
            self.base_position = ep_data['base_position']
            self.obstacles = ep_data['obstacles']
            self.obstacle_heights = ep_data['obstacle_heights']
            self.data_rates = ep_data['data_rates']
            self.rewards = ep_data['rewards']
            self.altitudes = ep_data['altitudes']
            self.speeds = ep_data['speeds']
            self.cumulative_distances = ep_data['cumulative_distances']
            
            # Create static plot
            print(f"      [→] Creating static plot for Episode {ep_idx+1}...")
            save_path = os.path.join(output_dir, f"episode_{ep_idx+1}_static.png")
            self.create_static_plot_enhanced(save_path, ep_idx + 1)
            
            # Create 2D animation
            save_path_2d = os.path.join(output_dir, f"episode_{ep_idx+1}_2d.gif")
            self.create_2d_animation(save_path_2d, ep_idx + 1)
            
            # Create 3D animation
            save_path_3d = os.path.join(output_dir, f"episode_{ep_idx+1}_3d.gif")
            self.create_3d_animation(save_path_3d, ep_idx + 1)
    
    def create_static_plot_enhanced(self, save_path, episode_num):
        """
        Create enhanced static plot with varied obstacle heights visualization.
        """
        fig = plt.figure(figsize=(20, 16))
        
        # --- 3D View with varied height obstacles ---
        ax1 = fig.add_subplot(3, 2, 1, projection='3d')
        ax1.set_title(f'Episode {episode_num} - 3D Trajectory (Colored by Speed)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_zlim(0, SPACE_Z)
        
        # Draw obstacles as cylinders with different heights
        for obs, height in zip(self.obstacles, self.obstacle_heights):
            theta = np.linspace(0, 2*np.pi, 20)
            radius = self.env.obstacle_radius
            x_circle = obs[0] + radius * np.cos(theta)
            y_circle = obs[1] + radius * np.sin(theta)
            
            # Draw cylinder sides
            for j in range(len(theta)-1):
                xs = [x_circle[j], x_circle[j+1], x_circle[j+1], x_circle[j]]
                ys = [y_circle[j], y_circle[j+1], y_circle[j+1], y_circle[j]]
                zs = [0, 0, height, height]
                verts = [list(zip(xs, ys, zs))]
                ax1.add_collection3d(Poly3DCollection(verts, alpha=0.3, 
                                                     facecolor='red', edgecolor='darkred'))
            
            # Top cap
            z_top = np.full_like(theta, height)
            verts_top = [list(zip(x_circle, y_circle, z_top))]
            ax1.add_collection3d(Poly3DCollection(verts_top, alpha=0.4, facecolor='red'))
            
            # Height label
            ax1.text(obs[0], obs[1], height + 5, f'{height:.0f}m', 
                    fontsize=9, ha='center', color='darkred', fontweight='bold')
        
        # Plot trajectory colored by speed
        max_speed = max(self.speeds) if self.speeds and max(self.speeds) > 0 else 1
        colors = plt.cm.plasma(np.array(self.speeds) / max_speed)
        
        for i in range(len(self.uav_positions) - 1):
            ax1.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    [self.uav_positions[i][2], self.uav_positions[i+1][2]],
                    color=colors[i], linewidth=2.5)
        
        ax1.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                   c='green', marker='s', s=200, label='Base', edgecolors='black', linewidths=2)
        ax1.legend()
        ax1.view_init(elev=25, azim=45)
        
        # --- Top View ---
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_xlim(0, SPACE_X)
        ax2.set_ylim(0, SPACE_Y)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        for i in range(len(self.uav_positions) - 1):
            ax2.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    color=colors[i], linewidth=2)
        
        for obs, height in zip(self.obstacles, self.obstacle_heights):
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax2.add_patch(circle)
            ax2.text(obs[0], obs[1], f'{height:.0f}m', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        ax2.plot(self.base_position[0], self.base_position[1], 'gs', 
                markersize=15, markeredgecolor='black', markeredgewidth=2)
        
        # --- Data Rate & Altitude ---
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.set_title('Data Rate & Altitude', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Data Rate (bps/Hz)', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3.grid(True, alpha=0.3)
        
        steps = list(range(len(self.data_rates)))
        ax3.plot(steps, self.data_rates, 'b-', linewidth=2, label='Data Rate')
        
        ax3_alt = ax3.twinx()
        ax3_alt.set_ylabel('Altitude (m)', color='orange')
        ax3_alt.tick_params(axis='y', labelcolor='orange')
        ax3_alt.plot(steps, self.altitudes, 'orange', linewidth=2, label='Altitude')
        
        # Add obstacle height reference lines
        for height in self.obstacle_heights:
            ax3_alt.axhline(y=height, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # --- Speed ---
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.set_title('UAV Speed Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Speed (m/s)', color='green')
        ax4.tick_params(axis='y', labelcolor='green')
        ax4.grid(True, alpha=0.3)
        ax4.plot(steps, self.speeds, 'g-', linewidth=2, label='Speed')
        avg_speed = np.mean(self.speeds)
        ax4.axhline(y=avg_speed, color='red', linestyle='--', alpha=0.5, 
                   label=f'Avg: {avg_speed:.2f} m/s')
        ax4.legend(loc='upper right')
        
        # --- Cumulative Distance ---
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.set_title('Cumulative Distance Traveled', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Distance (m)', color='purple')
        ax5.tick_params(axis='y', labelcolor='purple')
        ax5.grid(True, alpha=0.3)
        ax5.plot(steps, self.cumulative_distances, 'purple', linewidth=2.5)
        final_dist = self.cumulative_distances[-1] if self.cumulative_distances else 0
        ax5.text(0.98, 0.02, f'Final: {final_dist:.2f} m', transform=ax5.transAxes,
                fontsize=11, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # --- Statistics ---
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')
        ax6.set_title(f'Episode {episode_num} Statistics', fontsize=12, fontweight='bold')
        
        stats_text = (
            f'╔══════════════════════════════════════╗\n'
            f'║   EPISODE {episode_num} STATISTICS              ║\n'
            f'╚══════════════════════════════════════╝\n\n'
            f'🏔️  OBSTACLES:\n'
            f'  • Heights: {", ".join([f"{h:.0f}m" for h in self.obstacle_heights])}\n\n'
            f'📏 DISTANCE & SPEED:\n'
            f'  • Total Distance:  {final_dist:8.2f} m\n'
            f'  • Avg Speed:       {np.mean(self.speeds):8.2f} m/s\n'
            f'  • Max Speed:       {np.max(self.speeds):8.2f} m/s\n\n'
            f'📡 DATA RATE:\n'
            f'  • Avg Data Rate:   {np.mean(self.data_rates):8.2f} bps/Hz\n'
            f'  • Max Data Rate:   {np.max(self.data_rates):8.2f} bps/Hz\n\n'
            f'✈️  ALTITUDE:\n'
            f'  • Avg Altitude:    {np.mean(self.altitudes):8.1f} m\n'
            f'  • Max Altitude:    {np.max(self.altitudes):8.1f} m\n\n'
            f'🎯 REWARDS:\n'
            f'  • Total Reward:    {np.sum(self.rewards):8.1f}\n'
            f'  • Avg Reward:      {np.mean(self.rewards):8.2f}\n'
        )
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=9.5, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Episode {episode_num} plot saved")
        plt.close()
    
    def visualize_all(self, output_dir="visualizations"):
        """
        Main visualization function - collects data and creates all plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n╔{'═'*68}╗")
        print(f"║  ENHANCED UAV TRAJECTORY VISUALIZATION - MULTI-EPISODE           ║")
        print(f"╚{'═'*68}╝\n")
        
        # Collect data from all episodes
        for episode in range(self.num_episodes):
            self.collect_trajectory(episode)
        
        # Create comparison plot across all episodes
        self.create_multi_episode_comparison(
            save_path=os.path.join(output_dir, "multi_episode_comparison.png"))
        
        # Create individual detailed plots for each episode
        self.create_individual_episode_plots(output_dir)
        
        print(f"\n{'='*70}")
        print(f"✓ All visualizations saved to '{output_dir}/' directory")
        print(f"\n  MULTI-EPISODE FILES:")
        print(f"  └─ multi_episode_comparison.png")
        print(f"\n  PER-EPISODE FILES (for each episode):")
        print(f"  ├─ episode_N_static.png  (detailed static plot)")
        print(f"  ├─ episode_N_2d.gif      (2D animated trajectory)")
        print(f"  └─ episode_N_3d.gif      (3D animated trajectory)")
        print(f"{'='*70}\n")
        
        self.env.close()

if __name__ == "__main__":
    # Update this path to match your trained model
    MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\fyp RL\models\uav_ppo\1769524969\final_model_continuous.zip"
    
    if os.path.exists(MODEL_PATH):
        # Visualize multiple episodes (3-5 recommended for good comparison)
        visualizer = UAVTrajectoryVisualizer(
            model_path=MODEL_PATH, 
            num_episodes=3,  # Change this to visualize more/fewer episodes
            max_steps=200
        )
        visualizer.visualize_all(output_dir="visualizations_multi_episode")
        print("Done! Check the 'visualizations_multi_episode' folder for outputs.")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model.")
