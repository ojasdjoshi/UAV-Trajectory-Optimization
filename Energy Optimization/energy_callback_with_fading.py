import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class EnergyCallback(BaseCallback):
    """
    A custom callback to calculate and log energy metrics per episode.
    Tracks:
    - Average data rate per episode
    - Total UAV energy per episode
    - Total GU energy per episode
    - Weighted total energy per episode
    - QoS satisfaction rate
    - Fading statistics (shadowing and small-scale fading)
    """
    def __init__(self, verbose=0):
        super(EnergyCallback, self).__init__(verbose)
        # Per-episode storage
        self.episode_data_rates = []
        self.episode_uav_energies = []
        self.episode_gu_energies = []
        self.episode_weighted_energies = []
        self.episode_qos_met_count = []
        
        # Fading statistics
        self.episode_base_shadowing = []
        self.episode_user_shadowing = []
        self.episode_base_fading = []
        self.episode_user_fading = []
        
        self.episode_steps = 0
        
        # Episode counter
        self.episodes_finished = 0

    def _on_step(self):
        if isinstance(self.training_env, VecEnv):
            infos = self.locals.get('infos', [])
            
            for info in infos:
                # Accumulate metrics during the episode
                if 'data_rate' in info:
                    self.episode_data_rates.append(info['data_rate'])
                
                if 'uav_energy' in info:
                    self.episode_uav_energies.append(info['uav_energy'])
                
                if 'gu_energy' in info:
                    self.episode_gu_energies.append(info['gu_energy'])
                
                if 'weighted_energy' in info:
                    self.episode_weighted_energies.append(info['weighted_energy'])
                
                if 'qos_met' in info:
                    self.episode_qos_met_count.append(1 if info['qos_met'] else 0)
                
                # Fading information
                if 'fading' in info:
                    fading = info['fading']
                    self.episode_base_shadowing.append(fading.get('base_shadowing', 0))
                    self.episode_user_shadowing.append(fading.get('user_shadowing', 0))
                    self.episode_base_fading.append(fading.get('base_fading', 0))
                    self.episode_user_fading.append(fading.get('user_fading', 0))
                
                self.episode_steps += 1
                
                # Episode finished
                if 'terminal_observation' in info and info['terminal_observation'] is not None:
                    # Calculate and log episode metrics
                    if self.episode_data_rates:
                        avg_rate = np.mean(self.episode_data_rates)
                        self.logger.record('custom/average_data_rate', avg_rate)
                    
                    if self.episode_uav_energies:
                        total_uav_energy_kj = np.sum(self.episode_uav_energies) / 1000  # Convert J to kJ
                        self.logger.record('custom/total_uav_energy_kj', total_uav_energy_kj)
                    
                    if self.episode_gu_energies:
                        total_gu_energy_kj = np.sum(self.episode_gu_energies) / 1_000_000  # Convert mJ to kJ
                        self.logger.record('custom/total_gu_energy_kj', total_gu_energy_kj)
                    
                    if self.episode_weighted_energies:
                        avg_weighted_energy = np.mean(self.episode_weighted_energies)
                        total_weighted_energy = np.sum(self.episode_weighted_energies)
                        self.logger.record('custom/avg_weighted_energy', avg_weighted_energy)
                        self.logger.record('custom/total_weighted_energy', total_weighted_energy)
                    
                    if self.episode_qos_met_count:
                        qos_satisfaction_rate = np.mean(self.episode_qos_met_count) * 100  # Percentage
                        self.logger.record('custom/qos_satisfaction_rate', qos_satisfaction_rate)
                    
                    # Log energy efficiency (data rate per unit energy)
                    if self.episode_data_rates and self.episode_weighted_energies:
                        energy_efficiency = np.mean(self.episode_data_rates) / (np.mean(self.episode_weighted_energies) + 1e-6)
                        self.logger.record('custom/energy_efficiency', energy_efficiency)
                    
                    # Additional metrics
                    if self.episode_uav_energies and self.episode_gu_energies:
                        avg_uav_step = np.mean(self.episode_uav_energies)
                        avg_gu_step = np.mean(self.episode_gu_energies)
                        self.logger.record('custom/avg_uav_energy_per_step', avg_uav_step)
                        self.logger.record('custom/avg_gu_energy_per_step', avg_gu_step)
                    
                    # Fading statistics
                    if self.episode_base_shadowing:
                        avg_base_shadowing = np.mean(self.episode_base_shadowing)
                        std_base_shadowing = np.std(self.episode_base_shadowing)
                        self.logger.record('custom/avg_base_shadowing_db', avg_base_shadowing)
                        self.logger.record('custom/std_base_shadowing_db', std_base_shadowing)
                    
                    if self.episode_user_shadowing:
                        avg_user_shadowing = np.mean(self.episode_user_shadowing)
                        std_user_shadowing = np.std(self.episode_user_shadowing)
                        self.logger.record('custom/avg_user_shadowing_db', avg_user_shadowing)
                        self.logger.record('custom/std_user_shadowing_db', std_user_shadowing)
                    
                    if self.episode_base_fading:
                        avg_base_fading = np.mean(self.episode_base_fading)
                        std_base_fading = np.std(self.episode_base_fading)
                        self.logger.record('custom/avg_base_fading_db', avg_base_fading)
                        self.logger.record('custom/std_base_fading_db', std_base_fading)
                    
                    if self.episode_user_fading:
                        avg_user_fading = np.mean(self.episode_user_fading)
                        std_user_fading = np.std(self.episode_user_fading)
                        self.logger.record('custom/avg_user_fading_db', avg_user_fading)
                        self.logger.record('custom/std_user_fading_db', std_user_fading)
                    
                    if self.verbose > 0:
                        print(f"\n=== Episode {self.episodes_finished + 1} Summary ===")
                        print(f"Avg Data Rate: {avg_rate:.4f} bits/Hz")
                        print(f"Total UAV Energy: {total_uav_energy_kj:.2f} kJ")
                        print(f"Total GU Energy: {total_gu_energy_kj:.4f} kJ")
                        print(f"Total Weighted Energy: {total_weighted_energy:.4f} kJ")
                        print(f"QoS Satisfaction: {qos_satisfaction_rate:.1f}%")
                        print(f"Energy Efficiency: {energy_efficiency:.4f}")
                        if self.episode_base_shadowing:
                            print(f"Shadowing: Base={avg_base_shadowing:.2f}±{std_base_shadowing:.2f}dB, "
                                  f"User={avg_user_shadowing:.2f}±{std_user_shadowing:.2f}dB")
                        print(f"Episode Steps: {self.episode_steps}")
                        print("=" * 40 + "\n")
                    
                    # Reset storage for the next episode
                    self.episode_data_rates = []
                    self.episode_uav_energies = []
                    self.episode_gu_energies = []
                    self.episode_weighted_energies = []
                    self.episode_qos_met_count = []
                    self.episode_base_shadowing = []
                    self.episode_user_shadowing = []
                    self.episode_base_fading = []
                    self.episode_user_fading = []
                    self.episode_steps = 0
                    self.episodes_finished += 1
                    
        return True
