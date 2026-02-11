import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import DQN
from uavenv_discrete import UavEnv, SPACE_X, SPACE_Y, SPACE_Z, MIN_ALTITUDE, MAX_ALTITUDE
import os

class UAVTrajectoryVisualizer:
    """
    Visualizes UAV trajectory in real-time with 2D and 3D views for Discrete (DQN) models.
    Shows obstacles, user movement, performance metrics, velocity, and distance.
    """
    
    def __init__(self, model_path, num_episodes=1, max_steps=200):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained DQN model
            num_episodes: Number of episodes to visualize
            max_steps: Maximum steps per episode
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Initialize environment
        self.env = UavEnv()
        
        # Load the trained model - DQN for Discrete Action Space
        print(f"Loading DQN model from: {model_path}")
        self.model = DQN.load(model_path, env=self.env)
        
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
        self.obstacles = [(o.x, o.y, o.z) for o in self.env.obstacles]
        
        print(f"\n{'='*70}")
        print(f"  EPISODE {episode_num + 1}: TRAJECTORY COLLECTION (DQN)")
        print(f"{'='*70}")
        
        while not done and step < self.max_steps:
            # Predict the action from the trained DQN model (Discrete)
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
            
            # Store velocity and distance data from the info dict
            self.speeds.append(info.get('uav_speed', 0))
            self.velocities.append(info.get('uav_velocity', np.array([0, 0, 0])))
            self.step_distances.append(info.get('step_distance', 0))
            self.cumulative_distances.append(info.get('total_distance', 0))
            
            total_reward += reward
            step += 1
            
            # Display progress every 25 steps
            if step % 25 == 0:
                print(f"  Step {step:3d}/{self.max_steps} | "
                      f"Speed: {self.speeds[-1]:5.2f} m/s | "
                      f"Step Dist: {self.step_distances[-1]:5.2f} m | "
                      f"Total Dist: {self.cumulative_distances[-1]:7.2f} m | "
                      f"Data Rate: {self.data_rates[-1]:5.2f} bps/Hz")
        
        # Calculate statistics
        avg_speed = np.mean(self.speeds)
        max_speed = np.max(self.speeds)
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
        print(f"  └─ Maximum Speed:     {max_speed:.2f} m/s")
        print(f"\n  DATA RATE METRICS:")
        print(f"  ├─ Average Data Rate: {avg_data_rate:.2f} bps/Hz")
        print(f"  └─ Max Data Rate:     {np.max(self.data_rates):.2f} bps/Hz")
        print(f"\n  STATUS:")
        print(f"  └─ Crashed:           {'YES ❌' if terminated and step < self.max_steps else 'NO ✓'}")
        print(f"{'='*70}\n")
        
        return step
    
    def create_2d_animation(self, save_path="uav_trajectory_2d.gif", interval=100):
        """Create animated 2D top-down view with subplots."""
        print(f"  [→] Creating 2D animation...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- Top Left: Top-down view ---
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y)
        ax1.set_xlabel('X Position (m)'); ax1.set_ylabel('Y Position (m)')
        ax1.set_title('UAV Trajectory - Top View (DQN)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        ax1.plot(self.base_position[0], self.base_position[1], 'gs', markersize=15, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.3, label='Obstacle' if obs == self.obstacles[0] else '')
            ax1.add_patch(circle)
            ax1.plot(obs[0], obs[1], 'rx', markersize=10, markeredgewidth=2)
        
        uav_trail, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
        user_point, = ax1.plot([], [], 'mo', markersize=10, label='User', markeredgecolor='purple', markeredgewidth=2)
        ax1.legend(loc='upper right', fontsize=9)
        
        # --- Top Right: Data Rate & Altitude ---
        ax2.set_xlim(0, len(self.uav_positions))
        ax2.set_ylim(0, max(max(self.data_rates) * 1.1, 10))
        ax2.set_ylabel('Data Rate', color='blue')
        ax2.grid(True, alpha=0.3)
        data_rate_line, = ax2.plot([], [], 'b-', linewidth=2)
        
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylim(0, SPACE_Z)
        ax2_alt.set_ylabel('Altitude (m)', color='orange')
        altitude_line, = ax2_alt.plot([], [], 'orange', linewidth=2)
        
        # --- Bottom Left: Speed ---
        ax3.set_xlim(0, len(self.uav_positions))
        ax3.set_ylim(0, max(max(self.speeds) * 1.1, 1))
        ax3.set_ylabel('Speed (m/s)', color='green')
        ax3.grid(True, alpha=0.3)
        speed_line, = ax3.plot([], [], 'g-', linewidth=2)
        
        # --- Bottom Right: Distance ---
        ax4.set_xlim(0, len(self.uav_positions))
        ax4.set_ylim(0, max(self.cumulative_distances) * 1.1 if self.cumulative_distances else 1)
        ax4.set_ylabel('Distance (m)', color='purple')
        ax4.grid(True, alpha=0.3)
        distance_line, = ax4.plot([], [], 'purple', linewidth=2.5)
        
        step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        def init():
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
            
            step_text.set_text(f'Step: {frame+1}\nSpeed: {self.speeds[frame]:.2f} m/s\nDist: {self.cumulative_distances[frame]:.2f} m')
            return uav_trail, uav_point, user_point, data_rate_line, altitude_line, speed_line, distance_line, step_text
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.uav_positions), interval=interval, blit=True)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()

    def create_3d_animation(self, save_path="uav_trajectory_3d.gif", interval=100):
        """Create animated 3D view with rotating camera and velocity vectors."""
        print(f"  [→] Creating 3D animation...")
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, SPACE_X); ax.set_ylim(0, SPACE_Y); ax.set_zlim(0, SPACE_Z)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('UAV 3D Trajectory - DQN Agent', fontsize=14, fontweight='bold', pad=20)
        
        ax.scatter([self.base_position[0]], [self.base_position[1]], [self.base_position[2]], c='green', marker='s', s=200, label='Base Station', edgecolors='black', linewidths=2)
        
        for obs in self.obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(0, obs[2], 2)
            Theta, Z_cyl = np.meshgrid(theta, z_cyl)
            X_cyl = obs[0] + self.env.obstacle_radius * np.cos(Theta)
            Y_cyl = obs[1] + self.env.obstacle_radius * np.sin(Theta)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='red')
        
        uav_trail, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7)
        uav_point = ax.scatter([], [], [], c='blue', marker='o', s=150, edgecolors='darkblue', linewidths=2)
        user_point = ax.scatter([], [], [], c='magenta', marker='o', s=100, edgecolors='purple', linewidths=2)
        
        velocity_arrow = None
        step_text = fig.text(0.02, 0.95, '', fontsize=10, family='monospace', bbox=dict(facecolor='wheat', alpha=0.9))
        
        def animate(frame):
            nonlocal velocity_arrow
            x_t = [p[0] for p in self.uav_positions[:frame+1]]
            y_t = [p[1] for p in self.uav_positions[:frame+1]]
            z_t = [p[2] for p in self.uav_positions[:frame+1]]
            uav_trail.set_data(x_t, y_t)
            uav_trail.set_3d_properties(z_t)
            
            uav_point._offsets3d = ([self.uav_positions[frame][0]], [self.uav_positions[frame][1]], [self.uav_positions[frame][2]])
            user_point._offsets3d = ([self.user_positions[frame][0]], [self.user_positions[frame][1]], [self.user_positions[frame][2]])
            
            if velocity_arrow: velocity_arrow.remove()
            vel = self.velocities[frame]
            velocity_arrow = ax.quiver(self.uav_positions[frame][0], self.uav_positions[frame][1], self.uav_positions[frame][2], vel[0]*5, vel[1]*5, vel[2]*5, color='red', alpha=0.7)
            
            step_text.set_text(f'Step: {frame+1}\nSpeed: {self.speeds[frame]:.2f} m/s\nDist: {self.cumulative_distances[frame]:.2f} m')
            ax.view_init(elev=20, azim=45 + frame * 0.5)
            return uav_trail, uav_point, user_point, step_text
        
        anim = FuncAnimation(fig, animate, frames=len(self.uav_positions), interval=interval)
        anim.save(save_path, writer=PillowWriter(fps=10))
        plt.close()

    def create_static_plot(self, save_path="uav_trajectory_static.png"):
        """Create a comprehensive 6-panel static summary plot."""
        print(f"  [→] Creating static trajectory plot...")
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 3D View
        ax1 = fig.add_subplot(3, 2, 1, projection='3d')
        ax1.set_title('3D Trajectory View', fontweight='bold')
        ax1.set_xlim(0, SPACE_X); ax1.set_ylim(0, SPACE_Y); ax1.set_zlim(0, SPACE_Z)
        max_s = max(self.speeds) if self.speeds else 1
        colors = plt.cm.plasma(np.array(self.speeds) / max_s)
        for i in range(len(self.uav_positions)-1):
            ax1.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]], [self.uav_positions[i][1], self.uav_positions[i+1][1]], [self.uav_positions[i][2], self.uav_positions[i+1][2]], color=colors[i], linewidth=2)
        
        # 2. Top View
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.set_title('Top View (X-Y Plane)', fontweight='bold')
        ax2.set_xlim(0, SPACE_X); ax2.set_ylim(0, SPACE_Y); ax2.set_aspect('equal')
        for obs in self.obstacles:
            ax2.add_patch(Circle((obs[0], obs[1]), self.env.obstacle_radius, color='red', alpha=0.3))
        for i in range(len(self.uav_positions)-1):
            ax2.plot([self.uav_positions[i][0], self.uav_positions[i+1][0]], [self.uav_positions[i][1], self.uav_positions[i+1][1]], color=colors[i], linewidth=2)
        
        # 3. Data Rate & Alt
        ax3 = fig.add_subplot(3, 2, 3); steps = list(range(len(self.data_rates)))
        ax3.plot(steps, self.data_rates, 'b-', label='Rate'); ax3.set_ylabel('Rate', color='blue')
        ax3a = ax3.twinx(); ax3a.plot(steps, self.altitudes, 'orange', label='Alt'); ax3a.set_ylabel('Alt', color='orange')
        
        # 4. Speed
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(steps, self.speeds, 'g-'); ax4.set_ylabel('Speed (m/s)', color='green')
        
        # 5. Distance
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(steps, self.cumulative_distances, 'purple', linewidth=2); ax5.set_ylabel('Dist (m)', color='purple')
        
        # 6. Stats Panel
        ax6 = fig.add_subplot(3, 2, 6); ax6.axis('off')
        stats = (f'EPISODE STATISTICS (DQN)\n{"="*25}\n'
                 f'Total Distance: {self.cumulative_distances[-1]:.2f} m\n'
                 f'Avg Data Rate:  {np.mean(self.data_rates):.2f} bps/Hz\n'
                 f'Avg Speed:      {np.mean(self.speeds):.2f} m/s\n'
                 f'Max Altitude:   {np.max(self.altitudes):.1f} m\n'
                 f'Total Reward:   {np.sum(self.rewards):.1f}')
        ax6.text(0.1, 0.9, stats, verticalalignment='top', family='monospace', bbox=dict(facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

    def visualize_all(self, output_dir="visualizations_dqn"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n╔{'═'*68}╗\n║  UAV TRAJECTORY VISUALIZATION (DQN DISCRETE)                  ║\n╚{'═'*68}╝\n")
        for episode in range(self.num_episodes):
            self.collect_trajectory(episode)
            suf = f"_ep{episode+1}" if self.num_episodes > 1 else ""
            self.create_static_plot(os.path.join(output_dir, f"static{suf}.png"))
            self.create_2d_animation(os.path.join(output_dir, f"2d{suf}.gif"))
            self.create_3d_animation(os.path.join(output_dir, f"3d{suf}.gif"))
        self.env.close()

if __name__ == "__main__":
    MODEL = r"models\uav_dqn\1770443879\final_model_discrete.zip"
    if os.path.exists(MODEL):
        UAVTrajectoryVisualizer(model_path=MODEL, num_episodes=1).visualize_all()
    else:
        print(f"Error: Model not found at {MODEL}")