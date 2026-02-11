import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import warnings

class DataRateCallback(BaseCallback):
    """
    A custom callback to calculate and log the average data rate, velocity, 
    and distance traveled per episode in TensorBoard.
    """
    def __init__(self, verbose=0):
        super(DataRateCallback, self).__init__(verbose)
        self.episode_data_rates = []
        self.episode_speeds = []
        self.episode_distances = []
        self.episodes_finished = 0

    def _on_step(self):
        # Check if the environment is a vectorized environment (Monitor wraps it as VecEnv)
        if isinstance(self.training_env, VecEnv):
            infos = self.locals.get('infos', [])
            
            for info in infos:
                # Store the instantaneous metrics
                if 'data_rate' in info:
                    self.episode_data_rates.append(info['data_rate'])
                
                if 'uav_speed' in info:
                    self.episode_speeds.append(info['uav_speed'])
                
                if 'step_distance' in info:
                    self.episode_distances.append(info['step_distance'])
                
                # 'terminal_observation' is added by the Monitor wrapper upon episode end
                if 'terminal_observation' in info and info['terminal_observation'] is not None:
                    # Log the averages and totals for the just-finished episode
                    if self.episode_data_rates:
                        avg_rate = np.mean(self.episode_data_rates)
                        self.logger.record('custom/average_data_rate', avg_rate)
                    
                    if self.episode_speeds:
                        avg_speed = np.mean(self.episode_speeds)
                        max_speed = np.max(self.episode_speeds)
                        self.logger.record('custom/average_speed', avg_speed)
                        self.logger.record('custom/max_speed', max_speed)
                    
                    if self.episode_distances:
                        total_distance = np.sum(self.episode_distances)
                        avg_step_distance = np.mean(self.episode_distances)
                        self.logger.record('custom/total_distance', total_distance)
                        self.logger.record('custom/avg_step_distance', avg_step_distance)
                    
                    # Get episode distance from last info if available
                    if 'episode_distance' in info:
                        self.logger.record('custom/episode_distance', info['episode_distance'])
                    
                    # PRINTING: Only print if verbose is explicitly set > 0
                    if self.verbose > 0:
                        print(f"\nEpisode {self.episodes_finished + 1} Summary:")
                        print(f"  Avg Data Rate: {avg_rate:.4f} bps/Hz")
                        if self.episode_speeds:
                            print(f"  Avg Speed: {avg_speed:.4f} m/s")
                            print(f"  Max Speed: {max_speed:.4f} m/s")
                        if self.episode_distances:
                            print(f"  Total Distance: {total_distance:.2f} m")
                            print(f"  Avg Step Distance: {avg_step_distance:.4f} m")

                    # Reset storage for the next episode
                    self.episode_data_rates = []
                    self.episode_speeds = []
                    self.episode_distances = []
                    self.episodes_finished += 1
        return True