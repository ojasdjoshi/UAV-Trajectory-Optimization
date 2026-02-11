import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from uavenv import UavEnv, SPACE_X, SPACE_Y, SPACE_Z, MIN_ALTITUDE, MAX_ALTITUDE, NUM_USERS
import os

class UAVTrajectoryVisualizer:
    """
    Visualizes UAV trajectory in real-time with 2D and 3D views.
    Shows obstacles with variable heights, user movement, and performance metrics for multiple users.
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
        """Reset all trajectory tracking variables for multiple users."""
        self.uav_positions = []
        self.user_positions = [[] for _ in range(NUM_USERS)]
        self.base_position = None
        self.obstacles = []
        self.data_rates = []
        self.rewards = []
        self.altitudes = []
        
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
        
        # Store initial positions (including obstacle heights)
        self.base_position = (self.env.base.x, self.env.base.y, self.env.base.z)
        self.obstacles = [(o.x, o.y, o.z) for o in self.env.obstacles]
        
        print(f"\n{'='*60}")
        print(f"Episode {episode_num + 1}: Collecting trajectory data...")
        print(f"{'='*60}")
        
        while not done and step < self.max_steps:
            # Predict the action from the trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            self.uav_positions.append((self.env.uav.x, self.env.uav.y, self.env.uav.z))
            
            # Store positions for all 5 users
            for i in range(NUM_USERS):
                self.user_positions[i].append((self.env.users[i].x, self.env.users[i].y, self.env.users[i].z))
                
            self.data_rates.append(info.get('data_rate', 0))
            self.rewards.append(reward)
            self.altitudes.append(self.env.uav.z)
            
            total_reward += reward
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}/{self.max_steps} | "
                      f"Data Rate: {self.data_rates[-1]:.2f} bps/Hz | "
                      f"Altitude: {self.altitudes[-1]:.1f}m")
        
        print(f"\nEpisode Summary:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Data Rate: {np.mean(self.data_rates):.2f} bps/Hz")
        print(f"  Crash: {'Yes' if terminated and step < self.max_steps else 'No'}")
        
        return step
    
    def create_2d_animation(self, save_path="uav_trajectory_2d.gif", interval=100):
        """
        Create animated 2D top-down view of the trajectory with split screen metrics.
        """
        print(f"\nCreating 2D animation: {save_path}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # --- Left subplot: Top-down view ---
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
        
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, 
                          color='red', alpha=0.3, label='Obstacle' if obs == self.obstacles[0] else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'rx', markersize=10, markeredgewidth=2)
        
        # Initialize dynamic elements
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        
        # Markers and Links for 5 Users
        user_points = []
        uav_to_user_links = []
        colors = ['magenta', 'purple', 'tab:pink', 'hotpink', 'deeppink']
        
        for i in range(NUM_USERS):
            up, = ax1.plot([], [], 'o', color=colors[i], markersize=10, 
                          label=f'User {i+1}' if i==0 else None, markeredgecolor='black', markeredgewidth=1)
            user_points.append(up)
            link, = ax1.plot([], [], '--', color=colors[i], linewidth=1.0, alpha=0.4)
            uav_to_user_links.append(link)
            
        uav_to_base, = ax1.plot([], [], 'g--', linewidth=1.5, alpha=0.5, label='UAV-Base Link')
        
        ax1.legend(loc='upper right', fontsize=9)
        
        # --- Right subplot: Metrics ---
        ax2.set_xlim(0, len(self.uav_positions))
        ax2.set_ylim(0, max(max(self.data_rates) * 1.1, 10))
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Data Rate (bps/Hz)', fontsize=11, color='blue')
        ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        data_rate_line, = ax2.plot([], [], 'b-', linewidth=2, label='Data Rate')
        
        # Secondary y-axis for altitude
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylim(0, SPACE_Z)
        ax2_alt.set_ylabel('Altitude (m)', fontsize=11, color='orange')
        ax2_alt.tick_params(axis='y', labelcolor='orange')
        ax2_alt.axhline(y=MIN_ALTITUDE, color='red', linestyle=':', alpha=0.5, label='Min Alt')
        ax2_alt.axhline(y=MAX_ALTITUDE, color='red', linestyle=':', alpha=0.5, label='Max Alt')
        
        altitude_line, = ax2_alt.plot([], [], 'orange', linewidth=2, label='Altitude')
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_alt.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # Text annotations
        step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            uav_trail.set_data([], [])
            uav_point.set_data([], [])
            for up in user_points: up.set_data([], [])
            for link in uav_to_user_links: link.set_data([], [])
            uav_to_base.set_data([], [])
            data_rate_line.set_data([], [])
            altitude_line.set_data([], [])
            step_text.set_text('')
            return [uav_trail, uav_point, uav_to_base, data_rate_line, altitude_line, step_text] + user_points + uav_to_user_links
        
        def animate(frame):
            # Update UAV trail and position
            x_trail = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_trail = [pos[1] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_trail, y_trail)
            uav_point.set_data([self.uav_positions[frame][0]], [self.uav_positions[frame][1]])
            
            # Update all 5 users
            for i in range(NUM_USERS):
                user_points[i].set_data([self.user_positions[i][frame][0]], [self.user_positions[i][frame][1]])
                uav_to_user_links[i].set_data(
                    [self.uav_positions[frame][0], self.user_positions[i][frame][0]],
                    [self.uav_positions[frame][1], self.user_positions[i][frame][1]]
                )
            
            # Update communication links to base
            uav_to_base.set_data(
                [self.uav_positions[frame][0], self.base_position[0]],
                [self.uav_positions[frame][1], self.base_position[1]]
            )
            
            # Update metrics
            steps = list(range(frame + 1))
            data_rate_line.set_data(steps, self.data_rates[:frame+1])
            altitude_line.set_data(steps, self.altitudes[:frame+1])
            
            # Update text
            step_text.set_text(
                f'Step: {frame + 1}/{len(self.uav_positions)}\n'
                f'Data Rate: {self.data_rates[frame]:.2f} bps/Hz\n'
                f'Altitude: {self.altitudes[frame]:.1f} m\n'
                f'Reward: {self.rewards[frame]:.1f}'
            )
            
            return [uav_trail, uav_point, uav_to_base, data_rate_line, altitude_line, step_text] + user_points + uav_to_user_links
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=True, repeat=True)
        
        # Save animation
        print(f"Saving 2D animation to {save_path}...")
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"✓ 2D animation saved successfully!")
        
        plt.close()
    
    def create_3d_animation(self, save_path="uav_trajectory_3d.gif", interval=100):
        """
        Create animated 3D view of the trajectory with cubic scaling and multiple users.
        """
        print(f"\nCreating 3D animation: {save_path}")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Explicitly set limits to show the full 500x500x150 space
        ax.set_xlim(0, SPACE_X)
        ax.set_ylim(0, SPACE_Y)
        ax.set_zlim(0, SPACE_Z)
        
        # Force the Z axis to look as big as X and Y
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title('UAV 3D Trajectory Visualization', fontsize=14, fontweight='bold', pad=20)
        
        # Plot static elements
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                  c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(0, obs[2], 2) # Use specific height from environment
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='red')
        
        # Altitude bounds
        ax.plot([0, SPACE_X], [0, 0], [MIN_ALTITUDE, MIN_ALTITUDE], 'r:', linewidth=1, alpha=0.5)
        ax.plot([0, SPACE_X], [0, 0], [MAX_ALTITUDE, MAX_ALTITUDE], 'r:', linewidth=1, alpha=0.5)
        
        # Initialize dynamic elements
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7, label='UAV Path')
        uav_point = ax.scatter([], [], [], c='blue', marker='o', s=150, edgecolors='darkblue', linewidths=2)
        
        # Multiple Users setup
        user_heads = []
        colors = ['magenta', 'purple', 'tab:pink', 'hotpink', 'deeppink']
        for i in range(NUM_USERS):
            uh = ax.scatter([], [], [], c=colors[i], marker='o', s=100, edgecolors='black', linewidths=1.5, label=f'User {i+1}' if i==0 else None)
            user_heads.append(uh)
        
        # Text annotation
        step_text = fig.text(0.02, 0.95, '', fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=10)
        
        def init():
            uav_trail.set_data([], [])
            uav_trail.set_3d_properties([])
            step_text.set_text('')
            return [uav_trail, step_text]
        
        def animate(frame):
            # Update UAV trail
            x_trail = [pos[0] for pos in self.uav_positions[:frame+1]]
            y_trail = [pos[1] for pos in self.uav_positions[:frame+1]]
            z_trail = [pos[2] for pos in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_trail, y_trail)
            uav_trail.set_3d_properties(z_trail)
            
            # Update UAV current position
            uav_point._offsets3d = ([self.uav_positions[frame][0]], 
                                    [self.uav_positions[frame][1]], 
                                    [self.uav_positions[frame][2]])
            
            # Update all 5 user positions
            for i in range(NUM_USERS):
                user_heads[i]._offsets3d = ([self.user_positions[i][frame][0]], 
                                            [self.user_positions[i][frame][1]], 
                                            [self.user_positions[i][frame][2]])
            
            # Update text
            step_text.set_text(
                f'Step: {frame + 1}/{len(self.uav_positions)} | '
                f'Data Rate: {self.data_rates[frame]:.2f} bps/Hz | '
                f'Altitude: {self.altitudes[frame]:.1f} m'
            )
            
            # Rotate view slightly for dynamic effect
            ax.view_init(elev=20, azim=45 + frame * 0.5)
            
            return [uav_trail, uav_point, step_text] + user_heads
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.uav_positions), 
                           interval=interval, blit=False, repeat=True)
        
        print(f"Saving 3D animation to {save_path}...")
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print(f"✓ 3D animation saved successfully!")
        
        plt.close()
    
    def create_static_plot(self, save_path="uav_trajectory_static.png"):
        """
        Create a comprehensive static plot with multiple views and all users.
        """
        print(f"\nCreating static trajectory plot: {save_path}")
        
        fig = plt.figure(figsize=(18, 12))
        
        # Color gradient based on data rate
        max_rate = max(self.data_rates) if self.data_rates and max(self.data_rates) > 0 else 1
        colors = plt.cm.viridis(np.array(self.data_rates) / max_rate)
        
        # --- 3D View ---
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_title('3D Trajectory View (Cube Scaling)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        
        # Full space and cube-like aspect
        ax1.set_xlim(0, SPACE_X)
        ax1.set_ylim(0, SPACE_Y)
        ax1.set_zlim(0, SPACE_Z)
        ax1.set_box_aspect([1, 1, 1])
        
        for i in range(len(self.uav_positions) - 1):
            ax1.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    [self.uav_positions[i][2], self.uav_positions[i+1][2]],
                    color=colors[i], linewidth=2)
        
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 20)
            z_cyl = np.linspace(0, obs[2], 2)
            T, Z = np.meshgrid(theta, z_cyl)
            ax1.plot_surface(obs[0]+30*np.cos(T), obs[1]+30*np.sin(T), Z, alpha=0.1, color='red')
            
        ax1.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], 
                   c='green', marker='s', s=200, label='Base')
        
        # Plot final positions of all 5 users
        for i in range(NUM_USERS):
            ax1.scatter([self.user_positions[i][-1][0]], [self.user_positions[i][-1][1]], [0], 
                       marker='X', s=100, edgecolors='black', label=f'User {i+1}' if i==0 else None)
        
        ax1.legend()
        
        # --- Top View ---
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.set_xlim(0, SPACE_X)
        ax2.set_ylim(0, SPACE_Y)
        ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
        
        for i in range(len(self.uav_positions) - 1):
            ax2.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][1], self.uav_positions[i+1][1]],
                    color=colors[i], linewidth=2)
        
        for obs in self.obstacles:
            ax2.add_patch(Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.2))
        
        # Final User positions top view
        for i in range(NUM_USERS):
            ax2.scatter([self.user_positions[i][-1][0]], [self.user_positions[i][-1][1]], marker='o', s=80, edgecolors='black')
        
        # --- Side View ---
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Side View (X-Z Plane)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (m)'); ax3.set_ylabel('Altitude (m)')
        ax3.set_xlim(0, SPACE_X)
        ax3.set_ylim(0, SPACE_Z) 
        ax3.grid(True, alpha=0.3)
        
        for i in range(len(self.uav_positions) - 1):
            ax3.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]],
                    [self.uav_positions[i][2], self.uav_positions[i+1][2]],
                    color=colors[i], linewidth=2)
        
        # --- Metrics Plot ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Performance Metrics Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Step'); ax4.set_ylabel('Sum Data Rate (bps/Hz)', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue'); ax4.grid(True, alpha=0.3)
        
        steps = list(range(len(self.data_rates)))
        ax4.plot(steps, self.data_rates, 'b-', linewidth=2, label='Sum Rate')
        
        ax4_alt = ax4.twinx()
        ax4_alt.set_ylabel('Altitude (m)', color='orange')
        ax4_alt.tick_params(axis='y', labelcolor='orange')
        ax4_alt.plot(steps, self.altitudes, 'orange', linewidth=2, label='Altitude')
        
        stats_text = (f'Statistics:\n'
                     f'Avg Sum Rate: {np.mean(self.data_rates):.2f} bps/Hz\n'
                     f'Max Sum Rate: {np.max(self.data_rates):.2f} bps/Hz\n'
                     f'Avg Altitude: {np.mean(self.altitudes):.1f} m\n'
                     f'Total Reward: {np.sum(self.rewards):.1f}')
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Static plot saved to {save_path}")
        plt.close()
    
    def visualize_all(self, output_dir="visualizations"):
        """
        Run complete visualization pipeline: collect data and create all plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'#'*60}")
        print(f"# UAV TRAJECTORY VISUALIZATION (Multi-User)")
        print(f"{'#'*60}")
        
        for episode in range(self.num_episodes):
            # Collect trajectory
            self.collect_trajectory(episode)
            
            # Create visualizations
            episode_suffix = f"_ep{episode+1}" if self.num_episodes > 1 else ""
            
            self.create_static_plot(
                save_path=os.path.join(output_dir, f"trajectory_static{episode_suffix}.png")
            )
            self.create_2d_animation(
                save_path=os.path.join(output_dir, f"trajectory_2d{episode_suffix}.gif"),
                interval=100
            )
            self.create_3d_animation(
                save_path=os.path.join(output_dir, f"trajectory_3d{episode_suffix}.gif"),
                interval=100
            )
        
        print(f"\n{'='*60}")
        print(f"✓ All multi-user visualizations saved to '{output_dir}/' directory")
        print(f"{'='*60}\n")
        
        self.env.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Ensure this path matches your trained multi-user model
    MODEL_PATH = r"models\uav_ppo\1769142679\final_model_continuous.zip"
    
    if os.path.exists(MODEL_PATH):
        visualizer = UAVTrajectoryVisualizer(
            model_path=MODEL_PATH, 
            num_episodes=1, 
            max_steps=200
        )
        visualizer.visualize_all(output_dir="visualizations")
        print("Done! Check the 'visualizations' folder for outputs.")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")