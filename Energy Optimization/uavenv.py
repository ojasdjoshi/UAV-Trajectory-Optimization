import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random

# --- Simulation Parameters ---
# Scenario Configuration
FREQUENCY_MHZ = 2400  # Operating frequency in MHz (e.g., 2.4 GHz)
ENVIRONMENT_TYPE = "Urban"  # Can be "Urban" or "Rural"
AVG_BUILDING_HEIGHT = 15 # Average building height in meters for Rural scenario

# --- Communication System Parameters ---
TX_POWER_DBM = 35  # Transmit power in dBm (3126 mW)
NOISE_POWER_DBM = -94.0 # Noise power in dBm (based on 20MHz bandwidth and 7dB Noise Figure)

# --- Fading Parameters ---
# Shadowing (Log-normal fading)
SHADOWING_STD_DB = 8.0  # Standard deviation for log-normal shadowing (typical value 6-10 dB)
SHADOWING_DECORRELATION_DISTANCE = 20.0  # meters - distance for shadowing to decorrelate

# Small-scale fading parameters
RICIAN_K_FACTOR_DB = 10.0  # Rician K-factor for LOS links (in dB, typical 6-15 dB for UAV)
ENABLE_SMALL_SCALE_FADING = True  # Set to False to disable fast fading
ENABLE_SHADOWING = True  # Set to False to disable shadowing

# Fading update rates
SMALL_SCALE_FADING_UPDATE_RATE = 1  # Update every step (fast fading)
SHADOWING_UPDATE_RATE = 5  # Update every 5 steps (slower variation)

# --- Energy Model Parameters (from paper) ---
# UAV Energy Parameters
UAV_ALTITUDE = 200  # meters (H from paper)
P_0 = 79.86  # Blade profile power constant (W)
P_i = 88.63  # Induced power constant (W)
v_b = 7.2  # Tip speed of rotor blade (m/s)
v_0 = 4.03  # Mean rotor-induced velocity in hover (m/s)
d_0 = 0.6  # Fuselage drag ratio
rho = 1.225  # Air density (kg/m³)
s = 0.05  # Rotor solidity
A = 0.503  # Rotor disc area (m²)

# GU Energy Parameters
GU_TX_POWER_MW = 200  # Maximum transmit power in mW (p^max from paper)

# Time Parameters
TIME_SLOT_DURATION = 1.0  # Δt in seconds
TOTAL_SERVICE_PERIOD = 200  # T time-slots (max_steps)

# Energy weights for objective function
OMEGA_0 = 0.3  # Weight for GU energy
OMEGA_1 = 0.7  # Weight for UAV energy

# QoS Parameters
R_REQ = 500  # Minimum required data rate in kbps (R^req from paper)
BANDWIDTH_MHZ = 20  # D^t_{n,k} bandwidth in MHz

# 3D Space Configuration
SPACE_X = 500  # meters
SPACE_Y = 500  # meters
SPACE_Z = 150  # meters
MIN_ALTITUDE = 25   # Min operational altitude for the UAV
MAX_ALTITUDE = 125  # Max operational altitude for the UAV
MAX_MOVE_DIST = 10.0 # Maximum displacement in meters per step for continuous action

class Entity:
    """Represents any object (UAV, User, Base Station, Obstacle) in 3D space."""
    def __init__(self, x=None, y=None, z=None, height=0):
        self.x = random.uniform(0, SPACE_X) if x is None else x
        self.y = random.uniform(0, SPACE_Y) if y is None else y
        self.z = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE) if z is None else z
        self.height = height # Vertical extent of the obstacle
        
        # Track previous position for velocity calculation
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z

    def __sub__(self, other):
        return np.array([self.x - other.x, self.y - other.y, self.z - other.z])

    def action(self, continuous_delta):
        """
        Moves the entity based on a continuous displacement vector [dx, dy, dz].
        continuous_delta: np.array of shape (3,) with values in range [-1, 1].
        """
        # Store previous position
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        
        # Scale the continuous input [-1, 1] to the maximum allowed step size (10m)
        dx = continuous_delta[0] * MAX_MOVE_DIST
        dy = continuous_delta[1] * MAX_MOVE_DIST
        dz = continuous_delta[2] * MAX_MOVE_DIST
        
        self.x = np.clip(self.x + dx, 0, SPACE_X)
        self.y = np.clip(self.y + dy, 0, SPACE_Y)
        self.z = np.clip(self.z + dz, MIN_ALTITUDE, MAX_ALTITUDE)
    
    def get_velocity(self, dt):
        """Calculate velocity magnitude based on position change."""
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        dz = self.z - self.prev_z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        return distance / dt if dt > 0 else 0


class FadingChannel:
    """
    Manages fading effects for a wireless link.
    Includes both small-scale fading (Rician/Rayleigh) and large-scale shadowing (log-normal).
    """
    def __init__(self, link_name=""):
        self.link_name = link_name
        self.shadowing_db = 0.0  # Current shadowing value in dB
        self.prev_position = None
        self.accumulated_distance = 0.0
        
    def update_shadowing(self, current_pos, is_los, force_update=False):
        """
        Update log-normal shadowing based on spatial correlation.
        Shadowing changes gradually as the entity moves.
        """
        if not ENABLE_SHADOWING:
            self.shadowing_db = 0.0
            return
        
        curr_pos_array = np.array([current_pos.x, current_pos.y, current_pos.z])
        if self.prev_position is None or force_update:
            # Initialize or force new shadowing value
            self.shadowing_db = np.random.normal(0, SHADOWING_STD_DB)
            self.prev_position = curr_pos_array
            self.accumulated_distance = 0.0
        else:
            # Calculate distance moved
            distance_moved = np.linalg.norm(curr_pos_array - self.prev_position)
            self.accumulated_distance += distance_moved
            
            # Update shadowing with spatial correlation
            if self.accumulated_distance >= SHADOWING_DECORRELATION_DISTANCE:
                # Completely decorrelated - generate new value
                self.shadowing_db = np.random.normal(0, SHADOWING_STD_DB)
                self.accumulated_distance = 0.0
            else:
                # Partially correlated - use exponential correlation model
                correlation_coef = np.exp(-distance_moved / SHADOWING_DECORRELATION_DISTANCE)
                
                # Autoregressive model: new_value = correlation * old_value + sqrt(1-correlation²) * noise
                noise_std = SHADOWING_STD_DB * np.sqrt(1 - correlation_coef**2)
                self.shadowing_db = correlation_coef * self.shadowing_db + np.random.normal(0, noise_std)
            
            self.prev_position = curr_pos_array
    
    def generate_small_scale_fading(self, is_los):
        """
        Generate small-scale fading coefficient.
        - LOS: Rician fading with K-factor
        - NLOS: Rayleigh fading
        
        Returns:
            Fading power gain in linear scale
        """
        if not ENABLE_SMALL_SCALE_FADING:
            return 1.0  # No fading
        
        if is_los:
            # Rician fading for LOS
            K_linear = 10**(RICIAN_K_FACTOR_DB / 10)
            
            # Rician fading: h = sqrt(K/(K+1)) + sqrt(1/(K+1)) * (h_real + j*h_imag)
            # LOS component (deterministic)
            los_component = np.sqrt(K_linear / (K_linear + 1))
            
            # Scattered component (Rayleigh)
            scatter_scale = np.sqrt(1 / (K_linear + 1))
            h_real = np.random.normal(0, 1/np.sqrt(2)) * scatter_scale
            h_imag = np.random.normal(0, 1/np.sqrt(2)) * scatter_scale
            
            # Total channel coefficient
            h_total_real = los_component + h_real
            h_total_imag = h_imag
            
            # Power gain = |h|²
            power_gain = h_total_real**2 + h_total_imag**2
        else:
            # Rayleigh fading for NLOS
            # h = h_real + j*h_imag, where both are N(0, 1/2)
            h_real = np.random.normal(0, 1/np.sqrt(2))
            h_imag = np.random.normal(0, 1/np.sqrt(2))
            
            # Power gain = |h|²
            power_gain = h_real**2 + h_imag**2
        
        return power_gain


def is_link_nlos(p1, p2, obstacles, obstacle_radius):
    """Checks if the line segment between p1 and p2 is intersected by any finite cylindrical obstacle."""
    for obs in obstacles:
        # Step 1: Horizontal distance check in XY plane
        line_vec_xy = np.array([p2.x - p1.x, p2.y - p1.y])
        p1_to_obs_xy = np.array([obs.x - p1.x, obs.y - p1.y])
        line_len_sq_xy = line_vec_xy[0]**2 + line_vec_xy[1]**2
        
        if line_len_sq_xy == 0:
            dist_sq_xy = p1_to_obs_xy[0]**2 + p1_to_obs_xy[1]**2
            if dist_sq_xy < obstacle_radius**2:
                # If they are vertically above/below each other, check if either is blocked by height
                if p1.z < obs.height or p2.z < obs.height: return True
            continue
        
        # Project p1_to_obs onto line_vec in 2D
        t = np.dot(p1_to_obs_xy, line_vec_xy) / line_len_sq_xy
        t = np.clip(t, 0, 1) # Closest point on the segment
        
        closest_point_xy = np.array([p1.x, p1.y]) + t * line_vec_xy
        dist_sq_xy = (obs.x - closest_point_xy[0])**2 + (obs.y - closest_point_xy[1])**2
        
        # Step 2: Verticality check. Is the 3D ray passing below the obstacle height?
        if dist_sq_xy < obstacle_radius**2:
            # Interpolate the height of the signal at the intersection point
            # z(t) = z1 + t * (z2 - z1)
            z_at_intersection = p1.z + t * (p2.z - p1.z)
            if z_at_intersection < obs.height:
                return True
    return False


def calculate_3gpp_path_loss(pos1, pos2, frequency_mhz, env_type, avg_bldg_h=15, is_nlos=False):
    """Calculates path loss in dB using appropriate LOS/NLOS models from 3GPP TR 36.777."""
    d_3D = math.sqrt(np.sum((pos1 - pos2)**2))
    if d_3D < 1.0: d_3D = 1.0 # Minimum validity for standard models

    if env_type == "Urban":
        if is_nlos:
            fc_ghz = frequency_mhz / 1000
            return 32.4 + 20 * math.log10(fc_ghz) + 30 * math.log10(d_3D)
        else:
            h_UAV = pos1.z
            PL_3GPP = 28.0 + 22 * math.log10(d_3D) + 20 * math.log10(frequency_mhz / 1000)
            # Altitude-dependent Correction Factor (Equation 3 in paper)
            CF = 0.00010005 * h_UAV**2 - 0.0286 * h_UAV + 10.5169
            return PL_3GPP + CF

    elif env_type == "Rural":
        d_2D = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
        pl_b = min(0.03 * avg_bldg_h**1.72, 10) * math.log10(d_3D) - min(0.044 * avg_bldg_h**1.72, 14.77)
        PL_3GPP = 20 * math.log10(40 * math.pi * d_3D * frequency_mhz / 3000) + pl_b
        if d_2D <= 4000:
            CF = 2.8359 * math.log10(100 / d_2D) + 13.2785 if d_2D > 0 else 13.2785
        else:
            CF = 3.9745 * math.log10(1000 / d_2D) + 13.9739 if d_2D > 0 else 13.9739
        return PL_3GPP + CF
    return 300.0


def calculate_total_path_loss_with_fading(pos1, pos2, frequency_mhz, env_type, 
                                          avg_bldg_h, is_nlos, fading_channel):
    """
    Calculate total path loss including large-scale path loss, shadowing, and small-scale fading.
    """
    # 1. Large-scale path loss (deterministic)
    path_loss_db = calculate_3gpp_path_loss(pos1, pos2, frequency_mhz, env_type, 
                                            avg_bldg_h, is_nlos)
    
    # 2. Shadowing (log-normal, slow variation)
    shadowing_db = fading_channel.shadowing_db
    
    # 3. Small-scale fading (fast variation)
    fading_power_gain = fading_channel.generate_small_scale_fading(not is_nlos)
    fading_db = 10 * np.log10(fading_power_gain) if fading_power_gain > 0 else -100
    
    # Total path loss = Large-scale PL + Shadowing - Small-scale fading gain
    total_loss_db = path_loss_db + shadowing_db - fading_db
    
    components = {
        'path_loss_db': path_loss_db,
        'shadowing_db': shadowing_db,
        'small_scale_fading_db': fading_db,
        'total_loss_db': total_loss_db
    }
    
    return total_loss_db, components


def calculate_uav_energy(velocity, dt):
    """
    Calculate UAV energy consumption based on velocity using Equation (9) from paper.
    E_n(v) = P_0(1 + 3v²/v_b²) + P_i*sqrt(1 + v⁴/(4v_0⁴) - v²/(2v_0²)) + (1/2)d_0*ρ*s*A*v³
    """
    v = velocity
    
    # Blade profile power component
    blade_profile = P_0 * (1 + 3 * (v**2) / (v_b**2))
    
    # Induced power component
    term_inside_sqrt = 1 + (v**4) / (4 * v_0**4) - (v**2) / (2 * v_0**2)
    term_inside_sqrt = max(0, term_inside_sqrt)
    induced = P_i * math.sqrt(term_inside_sqrt)
    
    # Parasitic power component
    parasitic = 0.5 * d_0 * rho * s * A * (v**3)
    
    # Total power in Watts
    total_power = blade_profile + induced + parasitic
    
    # Energy = Power × Time
    energy_joules = total_power * dt
    
    return energy_joules


def calculate_gu_energy(tx_power_mw, dt):
    """
    Calculate GU (Ground User) energy consumption based on transmit power.
    Using Equation (11): E^GU_total = sum over all time-slots of p^t_k
    """
    # Energy = Power × Time (in millijoules)
    energy_mj = tx_power_mw * dt
    return energy_mj


class UavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self):
        super(UavEnv, self).__init__()
        self.max_steps = TOTAL_SERVICE_PERIOD
        
        # --- UPGRADE: CONTINUOUS ACTION SPACE ---
        # Action is a 3D displacement vector [dx, dy, dz] in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space (13 features: vectors, pos, normalized energy and progress)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        
        # --- UPGRADE: 9 OBSTACLES WITH DIFFERENT HEIGHTS (3x3 Grid) ---
        self.obstacle_radius = 15.0
        self.obstacles = []
        grid_points = [125, 250, 375]
        # Heights are randomized between 40m and 100m to create a diverse urban skyline
        for gx in grid_points:
            for gy in grid_points:
                h_obs = random.uniform(40, 100) 
                self.obstacles.append(Entity(x=gx, y=gy, z=0, height=h_obs))
        
        # Entity instances
        self.base = Entity(x=SPACE_X//2, y=SPACE_Y//2, z=0)
        self.user = Entity(z=0)
        self.uav = Entity()
        
        # Energy and state tracking
        self.total_uav_energy = 0.0
        self.total_gu_energy = 0.0
        self.step_uav_energy = 0.0
        self.step_gu_energy = 0.0
        
        # Fading channel instances
        self.base_channel = FadingChannel("UAV-Base")
        self.user_channel = FadingChannel("UAV-User")

    def _get_obs(self):
        """Construct the 13-feature observation vector."""
        vec_to_user = (self.user - self.uav) / np.array([SPACE_X, SPACE_Y, SPACE_Z])
        vec_to_base = (self.base - self.uav) / np.array([SPACE_X, SPACE_Y, SPACE_Z])
        uav_pos_norm = np.array([
            self.uav.x / SPACE_X * 2 - 1,
            self.uav.y / SPACE_Y * 2 - 1,
            (self.uav.z - MIN_ALTITUDE) / (MAX_ALTITUDE - MIN_ALTITUDE) * 2 - 1
        ])
        
        # Dynamic metrics normalization
        step_uav_norm = np.tanh(self.step_uav_energy / 1000)
        step_gu_norm = np.tanh(self.step_gu_energy / 200)
        velocity_norm = np.tanh(self.uav.get_velocity(TIME_SLOT_DURATION) / 20)
        
        # Mission progress indicator
        progress = (2 * self.current_step / self.max_steps) - 1
        
        return np.concatenate([
            vec_to_user, 
            vec_to_base, 
            uav_pos_norm,
            [step_uav_norm, step_gu_norm, velocity_norm, progress]
        ]).astype(np.float32)

    def _calculate_metrics(self):
        """
        Executes the physical and communication simulation logic.
        Calculates NLOS, Path Loss, SNR, Rate, Energy, and Rewards.
        """
        # 1. Determine if links are NLOS due to obstacles (3D check)
        is_base_link_nlos = is_link_nlos(self.uav, self.base, self.obstacles, self.obstacle_radius)
        is_user_link_nlos = is_link_nlos(self.uav, self.user, self.obstacles, self.obstacle_radius)

        # 2. Update shadowing (log-normal slow variation)
        if self.current_step % SHADOWING_UPDATE_RATE == 0:
            self.base_channel.update_shadowing(self.uav, not is_base_link_nlos)
            self.user_channel.update_shadowing(self.uav, not is_user_link_nlos)

        # 3. Calculate path loss with fading for both links
        path_loss_base_db, base_components = calculate_total_path_loss_with_fading(
            self.uav, self.base, FREQUENCY_MHZ, ENVIRONMENT_TYPE, 
            AVG_BUILDING_HEIGHT, is_base_link_nlos, self.base_channel
        )
        path_loss_user_db, user_components = calculate_total_path_loss_with_fading(
            self.uav, self.user, FREQUENCY_MHZ, ENVIRONMENT_TYPE,
            AVG_BUILDING_HEIGHT, is_user_link_nlos, self.user_channel
        )

        # 4. Conversion to linear scale for network logic
        tx_power_mw = 10**(TX_POWER_DBM / 10)
        noise_power_mw = 10**(NOISE_POWER_DBM / 10)
        path_loss_base_linear = 10**(path_loss_base_db / 10)
        path_loss_user_linear = 10**(path_loss_user_db / 10)
        
        # 5. Calculate SNR
        snr_base = tx_power_mw / (path_loss_base_linear * noise_power_mw)
        snr_user = tx_power_mw / (path_loss_user_linear * noise_power_mw)

        # 6. Shannon Capacity calculation
        rate_base = math.log2(1 + snr_base)
        rate_user = math.log2(1 + snr_user)
        
        # 7. Relay Bottleneck: Rate is limited by the weakest link
        bottleneck_rate = min(rate_base, rate_user)
        bottleneck_rate_kbps = bottleneck_rate * BANDWIDTH_MHZ * 1000
        
        # 8. Propulsion and Transmission Energy
        uav_velocity = self.uav.get_velocity(TIME_SLOT_DURATION)
        uav_energy = calculate_uav_energy(uav_velocity, TIME_SLOT_DURATION)
        gu_energy = calculate_gu_energy(GU_TX_POWER_MW, TIME_SLOT_DURATION)
        
        # Update trackers
        self.step_uav_energy = uav_energy
        self.step_gu_energy = gu_energy
        self.total_uav_energy += uav_energy
        self.total_gu_energy += gu_energy
        
        # 9. Weighted Objective Function (J from paper)
        weighted_energy = OMEGA_0 * (gu_energy / 1_000_000) + OMEGA_1 * (uav_energy / 1000)
        
        # 10. Reward Shaping
        qos_met = bottleneck_rate_kbps >= R_REQ
        if qos_met:
            qos_reward = 10.0  # Positive bonus for successful service
        else:
            qos_deficit = (R_REQ - bottleneck_rate_kbps) / R_REQ
            qos_reward = -20.0 * qos_deficit  # Penalty scaled by missing throughput
        
        # Efficiency and Optimization components
        energy_penalty = -weighted_energy * 10.0
        rate_bonus = (bottleneck_rate - 2.0) * 2.0 
        
        reward = qos_reward + energy_penalty + rate_bonus + 0.1 # Base survival incentive
        
        # Safety/Altitude Constraints
        if not (MIN_ALTITUDE <= self.uav.z <= MAX_ALTITUDE):
            reward -= 50
            
        # Crash logic: 3D Check (XY proximity AND height check)
        crash_done = False
        for obs in self.obstacles:
            dist_xy = math.sqrt((self.uav.x - obs.x)**2 + (self.uav.y - obs.y)**2)
            if dist_xy < self.obstacle_radius:
                if self.uav.z < obs.height:
                    reward -= 200 # Heavy crash penalty
                    crash_done = True
                    break
        
        # Comprehensive Info Logs
        fading_info = {
            'base_path_loss': base_components['path_loss_db'],
            'base_shadowing': base_components['shadowing_db'],
            'base_fading': base_components['small_scale_fading_db'],
            'user_path_loss': user_components['path_loss_db'],
            'user_shadowing': user_components['shadowing_db'],
            'user_fading': user_components['small_scale_fading_db'],
        }
                
        return reward, bottleneck_rate, crash_done, uav_energy, gu_energy, weighted_energy, bottleneck_rate_kbps, fading_info

    def _get_reward(self):
        """Internal reward calculator."""
        reward, _, _, _, _, _, _, _ = self._calculate_metrics()
        return reward

    def reset(self, seed=None, options=None):
        """Initialize a new simulation episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        
        # Re-set locations
        self.base = Entity(x=SPACE_X//2, y=SPACE_Y//2, z=0)
        self.user = Entity(z=0)
        self.uav = Entity()
        
        # Reset trackers
        self.total_uav_energy = 0.0
        self.total_gu_energy = 0.0
        self.step_uav_energy = 0.0
        self.step_gu_energy = 0.0
        
        # Re-init fading channels
        self.base_channel = FadingChannel("UAV-Base")
        self.user_channel = FadingChannel("UAV-User")
        self.base_channel.update_shadowing(self.uav, True, force_update=True)
        self.user_channel.update_shadowing(self.uav, True, force_update=True)
        
        return self._get_obs(), {}

    def step(self, action):
        """Progress the environment by one second using a continuous control input."""
        self.current_step += 1
        
        # Apply the continuous [dx, dy, dz] action
        self.uav.action(action)
        
        # Ground user movement (slow pedestrian speed)
        if self.current_step % 20 == 0:
            dx, dy = random.choice([(10,0), (-10,0), (0,10), (0,-10)])
            self.user.x = np.clip(self.user.x + dx, 0, SPACE_X)
            self.user.y = np.clip(self.user.y + dy, 0, SPACE_Y)
            
        reward, rate, crash_done, uav_en, gu_en, w_en, rate_kbps, f_info = self._calculate_metrics()
        
        # Termination conditions
        self.done = crash_done or (self.current_step >= self.max_steps)
        
        info = {
            "data_rate": rate,
            "data_rate_kbps": rate_kbps,
            "uav_energy": uav_en,
            "gu_energy": gu_en,
            "weighted_energy": w_en,
            "total_uav_energy": self.total_uav_energy,
            "total_gu_energy": self.total_gu_energy,
            "qos_met": rate_kbps >= R_REQ,
            "fading": f_info
        }
        
        return self._get_obs(), reward, self.done, False, info

    def render(self):
        """Terminal-based simulation status."""
        reward, _, _, _, _, _, _, fading_info = self._calculate_metrics()
        print(f"Step: {self.current_step} | UAV: ({self.uav.x:.1f},{self.uav.y:.1f},{self.uav.z:.1f}) | "
              f"Reward: {reward:.2f}")

    def close(self):
        pass