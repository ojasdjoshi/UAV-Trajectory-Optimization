import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import warnings

class DataRateCallback(BaseCallback):
    """
    A custom callback to calculate and log the average data rate per episode 
    under the 'custom/average_data_rate' tag in TensorBoard.
    It expects the instantaneous data rate to be in the 'data_rate' key of the info dict.
    """
    def __init__(self, verbose=0):
        super(DataRateCallback, self).__init__(verbose)
        self.episode_data_rates = []
        self.episodes_finished = 0

    def _on_step(self):
        # Check if the environment is a vectorized environment (Monitor wraps it as VecEnv)
        if isinstance(self.training_env, VecEnv):
            infos = self.locals.get('infos', [])
            
            for info in infos:
                # Store the instantaneous data rate
                if 'data_rate' in info:
                    self.episode_data_rates.append(info['data_rate'])
                
                # 'terminal_observation' is added by the Monitor wrapper upon episode end
                if 'terminal_observation' in info and info['terminal_observation'] is not None:
                    # Log the average rate for the just-finished episode
                    if self.episode_data_rates:
                        avg_rate = np.mean(self.episode_data_rates)
                        
                        # Log to TensorBoard under the desired label
                        self.logger.record('custom/average_data_rate', avg_rate)
                        
                        # PRINTING REMOVED: Only print if verbose is explicitly set > 0
                        if self.verbose > 0:
                            print(f"Episode {self.episodes_finished + 1} Avg Data Rate: {avg_rate:.4f}")

                    # Reset storage for the next episode
                    self.episode_data_rates = []
                    self.episodes_finished += 1
        return True