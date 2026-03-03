import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random

# --- Simulation Parameters ---
FREQUENCY_MHZ = 2400
ENVIRONMENT_TYPE = "Urban" # Can be "Urban" or "Rural"
AVG_BUILDING_HEIGHT = 15

# Communication System Parameters
TX_POWER_DBM = 23.0
NOISE_POWER_DBM = -94.0

# 3D Space Configuration
SPACE_X = 500
SPACE_Y = 500
SPACE_Z = 150
MIN_ALTITUDE = 25
MAX_ALTITUDE = 125

NUM_USERS = 1 
MAX_UAV_SPEED = 10.0 # Max meters per step for continuous movement
MAX_USER_SPEED = 10.0 # Max meters per step for continuous user movement

class Entity:
    """Represents any object (UAV, User, Base Station, Obstacle) in 3D space."""
    def __init__(self, x=None, y=None, z=None):
        self.x = random.randint(0, SPACE_X) if x is None else x
        self.y = random.randint(0, SPACE_Y) if y is None else y
        self.z = random.randint(MIN_ALTITUDE, MAX_ALTITUDE) if z is None else z
        
        # NEW: Track velocity and distance
        self.vx = 0.0  # velocity in x direction (m/s)
        self.vy = 0.0  # velocity in y direction (m/s)
        self.vz = 0.0  # velocity in z direction (m/s)
        self.total_distance = 0.0  # cumulative distance traveled (m)

    def __sub__(self, other):
        return np.array([self.x - other.x, self.y - other.y, self.z - other.z])

    def action(self, continuous_action):
        """Moves the entity based on continuous velocity vectors [-1, 1]."""
        # Store previous position for distance calculation
        prev_x, prev_y, prev_z = self.x, self.y, self.z
        
        # Map normalized action [-1, 1] from the model to [-MAX_UAV_SPEED, MAX_UAV_SPEED]
        dx = float(continuous_action[0]) * MAX_UAV_SPEED
        dy = float(continuous_action[1]) * MAX_UAV_SPEED
        dz = float(continuous_action[2]) * MAX_UAV_SPEED

        self.x = np.clip(self.x + dx, 0, SPACE_X)
        self.y = np.clip(self.y + dy, 0, SPACE_Y)
        # Guard rail for UAV altitude
        self.z = np.clip(self.z + dz, MIN_ALTITUDE, MAX_ALTITUDE)
        
        # NEW: Calculate actual displacement and velocity
        actual_dx = self.x - prev_x
        actual_dy = self.y - prev_y
        actual_dz = self.z - prev_z
        
        # Store velocity (displacement per step)
        self.vx = actual_dx
        self.vy = actual_dy
        self.vz = actual_dz
        
        # Calculate 3D distance traveled this step
        step_distance = math.sqrt(actual_dx**2 + actual_dy**2 + actual_dz**2)
        self.total_distance += step_distance
        
        return step_distance  # Return distance traveled this step
    
    def get_speed(self):
        """Returns current speed (magnitude of velocity vector) in m/s"""
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def get_velocity_vector(self):
        """Returns velocity as [vx, vy, vz]"""
        return np.array([self.vx, self.vy, self.vz])
    
    def reset_distance(self):
        """Reset total distance counter"""
        self.total_distance = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

def is_link_nlos(p1, p2, obstacles, obstacle_radius):
    """Checks if the line segment between p1 and p2 is intersected by any cylindrical obstacle below its height."""
    for obs in obstacles:
        # 2D projection for intersection check
        line_vec = np.array([p2.x - p1.x, p2.y - p1.y])
        p1_to_obs = np.array([obs.x - p1.x, obs.y - p1.y])
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        if line_len_sq == 0:
            dist_sq = p1_to_obs[0]**2 + p1_to_obs[1]**2
            if dist_sq < obstacle_radius**2:
                # If vertical distance is below obstacle height
                if min(p1.z, p2.z) < obs.z: return True
            continue
            
        t = np.dot(p1_to_obs, line_vec) / line_len_sq
        t = np.clip(t, 0, 1) # Closest point on the line segment
        
        closest_point_2d = np.array([p1.x, p1.y]) + t * line_vec
        dist_sq = (obs.x - closest_point_2d[0])**2 + (obs.y - closest_point_2d[1])**2
        
        if dist_sq < obstacle_radius**2:
            # Check the Z-coordinate at the point of intersection
            z_at_intersection = p1.z + t * (p2.z - p1.z)
            if z_at_intersection < obs.z:
                return True
    return False

def calculate_3gpp_path_loss(pos1, pos2, frequency_mhz, env_type, avg_bldg_h=15, is_nlos=False):
    """Calculates path loss in dB using appropriate LOS/NLOS models."""
    d_3D = math.sqrt(np.sum((pos1 - pos2)**2))
    if d_3D == 0: return 300.0

    if env_type == "Urban":
        if is_nlos:
            fc_ghz = frequency_mhz / 1000
            return 32.4 + 20 * math.log10(fc_ghz) + 30 * math.log10(d_3D)
        else:
            h_UAV = pos1.z
            PL_3GPP = 28.0 + 22 * math.log10(d_3D) + 20 * math.log10(frequency_mhz / 1000)
            CF = 0.00010005 * h_UAV**2 - 0.0286 * h_UAV + 10.5169
            return PL_3GPP + CF

    elif env_type == "Rural":
        d_2D = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
        pl_b = min(0.03 * avg_bldg_h**1.72, 10) * math.log10(d_3D) - min(0.044 * avg_bldg_h**1.72, 14.77)
        PL_3GPP = 20 * math.log10(40 * math.pi * d_3D * frequency_mhz / 3000) + pl_b
        if d_2D <= 4000:
            CF = 2.8359 * math.log10(100 / d_2D) + 13.2785 if d_2D > 0 else 0
        else:
            CF = 3.9745 * math.log10(1000 / d_2D) + 13.9739 if d_2D > 0 else 0
        return PL_3GPP + CF
    return 300.0

def calculate_rician_k_factor(pos1, pos2, is_nlos):
    if is_nlos: return 0.1
    h_uav = pos1.z
    d_2D = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
    if d_2D > 0: elevation_angle = math.atan(h_uav / d_2D)
    else: elevation_angle = math.pi / 2
    
    K_db = 0 + 10 * math.sin(elevation_angle)
    return 10**(K_db / 10)

def generate_rician_fading(K_factor):
    los_component = np.sqrt(K_factor / (K_factor + 1))
    scatter_scale = np.sqrt(1 / (K_factor + 1))
    std_dev = 1/np.sqrt(2)
    h_real = np.random.normal(0, std_dev)
    h_imag = np.random.normal(0, std_dev)
    h = los_component + scatter_scale * (h_real + 1j * h_imag)
    return abs(h)**2

class UavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self):
        super(UavEnv, self).__init__()
        self.max_steps = 200
        
        # --- CONTINUOUS ACTION SPACE ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # --- OBSERVATION SPACE FOR 1 USER ---
        # [v_to_user(3), v_to_base(3), uav_pos_norm(3)] = 9
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        
        # 5 Obstacles with varying heights (Minimum 30m)
        self.obstacles = [
            Entity(x=125, y=250, z=60),   # Obs 1: Height 60m
            Entity(x=375, y=250, z=90),   # Obs 2: Height 90m
            Entity(x=250, y=125, z=40),   # Obs 3: Height 40m
            Entity(x=250, y=375, z=110),  # Obs 4: Height 110m
            Entity(x=100, y=100, z=130)   # Obs 5: Height 130m
        ]
        self.obstacle_radius = 30
        
        # Variables to hold data for logging
        self.last_sum_rate = 0.0
        self.last_user_rates = []
        
        # NEW: Episode statistics
        self.episode_total_distance = 0.0
        self.step_distance = 0.0

    def _get_obs(self):
        vec_to_user = (self.user - self.uav) / np.array([SPACE_X, SPACE_Y, SPACE_Z])
        vec_to_base = (self.base - self.uav) / np.array([SPACE_X, SPACE_Y, SPACE_Z])
        
        uav_pos_norm = np.array([
            self.uav.x / SPACE_X * 2 - 1,
            self.uav.y / SPACE_Y * 2 - 1,
            (self.uav.z - MIN_ALTITUDE) / (MAX_ALTITUDE - MIN_ALTITUDE) * 2 - 1
        ])
        
        return np.concatenate([vec_to_user, vec_to_base, uav_pos_norm]).astype(np.float32)

    def _get_reward(self):
        # 1. Backhaul Rate
        is_base_nlos = is_link_nlos(self.uav, self.base, self.obstacles, self.obstacle_radius)
        pl_base = calculate_3gpp_path_loss(self.uav, self.base, FREQUENCY_MHZ, ENVIRONMENT_TYPE, AVG_BUILDING_HEIGHT, is_nlos=is_base_nlos)
        k_base = calculate_rician_k_factor(self.uav, self.base, is_base_nlos)
        gain_base = generate_rician_fading(k_base)
        
        tx_mw = 10**(TX_POWER_DBM / 10)
        noise_mw = 10**(NOISE_POWER_DBM / 10)
        pl_base_lin = 10**(pl_base / 10)
        snr_base = (tx_mw * gain_base) / (pl_base_lin * noise_mw)
        rate_base = math.log2(1 + snr_base)

        # 2. User Rates (Only 1 user)
        is_u_nlos = is_link_nlos(self.uav, self.user, self.obstacles, self.obstacle_radius)
        pl_user = calculate_3gpp_path_loss(self.uav, self.user, FREQUENCY_MHZ, ENVIRONMENT_TYPE, AVG_BUILDING_HEIGHT, is_nlos=is_u_nlos)
        k_user = calculate_rician_k_factor(self.uav, self.user, is_u_nlos)
        gain_user = generate_rician_fading(k_user)
        
        pl_user_lin = 10**(pl_user / 10)
        snr_user = (tx_mw * gain_user) / (pl_user_lin * noise_mw)
        r_user = math.log2(1 + snr_user)

        self.last_user_rates = [r_user] 
        
        # 3. Sum Rate vs Backhaul
        final_sum_rate = min(r_user, rate_base)
        self.last_sum_rate = final_sum_rate

        # 4. Reward Calculation
        reward = final_sum_rate * 10 # Scale for better RL gradients
        
        # 5. Penalties
        if r_user < 1.0:
            reward -= 5 
        
        for obs in self.obstacles:
            dist = math.sqrt((self.uav.x - obs.x)**2 + (self.uav.y - obs.y)**2)
            # Crash only if within radius AND below obstacle height
            if dist < self.obstacle_radius and self.uav.z < obs.z:
                reward -= 1000
                self.done = True
                self.last_sum_rate = 0
                break
                
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        
        self.base = Entity(x=SPACE_X//2, y=SPACE_Y//2, z=0)
        self.uav = Entity()
        self.user = Entity(z=0)
        
        # NEW: Reset distance tracking
        self.uav.reset_distance()
        self.episode_total_distance = 0.0
        self.step_distance = 0.0
        
        self.last_sum_rate = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # NEW: Track distance traveled this step
        self.step_distance = self.uav.action(action)
        self.episode_total_distance += self.step_distance
        
        # Ground User Continuous Movement (Every 10 steps)
        if self.current_step % 10 == 0:
            # Generate random continuous displacement
            dx = random.uniform(-MAX_USER_SPEED, MAX_USER_SPEED)
            dy = random.uniform(-MAX_USER_SPEED, MAX_USER_SPEED)
            self.user.x = np.clip(self.user.x + dx, 0, SPACE_X)
            self.user.y = np.clip(self.user.y + dy, 0, SPACE_Y)
        
        reward = self._get_reward()
        terminated = self.done
        truncated = self.current_step >= self.max_steps
        
        # NEW: Enhanced info dictionary with velocity and distance
        info = {
            "data_rate": self.last_sum_rate,
            "altitude": self.uav.z,
            "uav_speed": self.uav.get_speed(),  # Current speed (m/s)
            "uav_velocity": self.uav.get_velocity_vector(),  # [vx, vy, vz]
            "step_distance": self.step_distance,  # Distance this step (m)
            "total_distance": self.uav.total_distance,  # Cumulative distance (m)
            "episode_distance": self.episode_total_distance  # Episode distance (m)
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        speed = self.uav.get_speed()
        print(f"Step: {self.current_step} | UAV: ({self.uav.x:.0f},{self.uav.y:.0f},{self.uav.z:.0f}) | "
              f"Speed: {speed:.2f} m/s | Distance: {self.step_distance:.2f} m | Reward: {self.last_sum_rate:.2f}")

    def close(self):
        pass