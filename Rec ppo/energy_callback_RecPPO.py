import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EnergyCallback(BaseCallback):
    """
    Custom callback for RecurrentPPO (sb3-contrib) — logs all metrics
    identical to the PPO and SAC EnergyCallbacks for apples-to-apples
    TensorBoard comparison.

    RecurrentPPO wraps the env in a VecEnv internally, so:
      - infos is always a list
      - dones is always an array
      - Episode end is detected via dones[i] == True

    Tracks per-episode:
        Communication : average data rate (bits/Hz), average data rate (kbps)
        Energy        : total UAV energy (kJ), total GU energy (kJ),
                        avg/total weighted energy, avg energy per step
        QoS           : satisfaction rate (%)
        Efficiency    : data rate per unit weighted energy
        Fading        : mean +/- std of base/user shadowing & small-scale fading (dB)
    """

    def __init__(self, verbose=0):
        super(EnergyCallback, self).__init__(verbose)

        # ── per-episode accumulators ──────────────────────────────────────────
        self.episode_data_rates        = []
        self.episode_data_rates_kbps   = []
        self.episode_uav_energies      = []
        self.episode_gu_energies       = []
        self.episode_weighted_energies = []
        self.episode_qos_met           = []

        # fading
        self.episode_base_shadowing = []
        self.episode_user_shadowing = []
        self.episode_base_fading    = []
        self.episode_user_fading    = []

        self.episode_steps     = 0
        self.episodes_finished = 0

    # ─────────────────────────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):

            # ── accumulate step-level metrics ─────────────────────────────────
            if "data_rate" in info:
                self.episode_data_rates.append(info["data_rate"])

            if "data_rate_kbps" in info:
                self.episode_data_rates_kbps.append(info["data_rate_kbps"])

            if "uav_energy" in info:
                self.episode_uav_energies.append(info["uav_energy"])

            if "gu_energy" in info:
                self.episode_gu_energies.append(info["gu_energy"])

            if "weighted_energy" in info:
                self.episode_weighted_energies.append(info["weighted_energy"])

            if "qos_met" in info:
                self.episode_qos_met.append(1 if info["qos_met"] else 0)

            if "fading" in info:
                f = info["fading"]
                self.episode_base_shadowing.append(f.get("base_shadowing", 0))
                self.episode_user_shadowing.append(f.get("user_shadowing", 0))
                self.episode_base_fading.append(f.get("base_fading", 0))
                self.episode_user_fading.append(f.get("user_fading", 0))

            self.episode_steps += 1

            # ── episode-end detection ─────────────────────────────────────────
            # RecurrentPPO uses VecEnv: episode end is signalled by dones[i]
            episode_done = i < len(dones) and bool(dones[i])

            if episode_done:
                self._log_episode_metrics()
                self._reset_episode_buffers()

        return True

    # ─────────────────────────────────────────────────────────────────────────
    def _log_episode_metrics(self):
        """Compute and record all metrics to TensorBoard."""

        # ── data rate ─────────────────────────────────────────────────────────
        avg_rate = 0.0
        if self.episode_data_rates:
            avg_rate = np.mean(self.episode_data_rates)
            self.logger.record("custom/average_data_rate", avg_rate)

        avg_rate_kbps = 0.0
        if self.episode_data_rates_kbps:
            avg_rate_kbps = np.mean(self.episode_data_rates_kbps)
            self.logger.record("custom/average_data_rate_kbps", avg_rate_kbps)

        # ── UAV energy ────────────────────────────────────────────────────────
        total_uav_energy_kj = avg_uav_step = 0.0
        if self.episode_uav_energies:
            total_uav_energy_kj = np.sum(self.episode_uav_energies) / 1_000      # J → kJ
            avg_uav_step        = np.mean(self.episode_uav_energies)
            self.logger.record("custom/total_uav_energy_kj",     total_uav_energy_kj)
            self.logger.record("custom/avg_uav_energy_per_step", avg_uav_step)

        # ── GU energy ─────────────────────────────────────────────────────────
        total_gu_energy_kj = avg_gu_step = 0.0
        if self.episode_gu_energies:
            total_gu_energy_kj = np.sum(self.episode_gu_energies) / 1_000_000    # mJ → kJ
            avg_gu_step        = np.mean(self.episode_gu_energies)
            self.logger.record("custom/total_gu_energy_kj",     total_gu_energy_kj)
            self.logger.record("custom/avg_gu_energy_per_step", avg_gu_step)

        # ── weighted energy ───────────────────────────────────────────────────
        avg_weighted_energy = total_weighted_energy = 0.0
        if self.episode_weighted_energies:
            avg_weighted_energy   = np.mean(self.episode_weighted_energies)
            total_weighted_energy = np.sum(self.episode_weighted_energies)
            self.logger.record("custom/avg_weighted_energy",   avg_weighted_energy)
            self.logger.record("custom/total_weighted_energy", total_weighted_energy)

        # ── QoS ───────────────────────────────────────────────────────────────
        qos_satisfaction_rate = 0.0
        if self.episode_qos_met:
            qos_satisfaction_rate = np.mean(self.episode_qos_met) * 100
            self.logger.record("custom/qos_satisfaction_rate", qos_satisfaction_rate)

        # ── energy efficiency ─────────────────────────────────────────────────
        energy_efficiency = 0.0
        if self.episode_data_rates and self.episode_weighted_energies:
            energy_efficiency = (
                np.mean(self.episode_data_rates)
                / (np.mean(self.episode_weighted_energies) + 1e-6)
            )
            self.logger.record("custom/energy_efficiency", energy_efficiency)

        # ── fading: base shadowing ────────────────────────────────────────────
        avg_bs = std_bs = 0.0
        if self.episode_base_shadowing:
            avg_bs = np.mean(self.episode_base_shadowing)
            std_bs = np.std(self.episode_base_shadowing)
            self.logger.record("custom/avg_base_shadowing_db", avg_bs)
            self.logger.record("custom/std_base_shadowing_db", std_bs)

        # ── fading: user shadowing ────────────────────────────────────────────
        avg_us = std_us = 0.0
        if self.episode_user_shadowing:
            avg_us = np.mean(self.episode_user_shadowing)
            std_us = np.std(self.episode_user_shadowing)
            self.logger.record("custom/avg_user_shadowing_db", avg_us)
            self.logger.record("custom/std_user_shadowing_db", std_us)

        # ── fading: base small-scale ──────────────────────────────────────────
        avg_bf = std_bf = 0.0
        if self.episode_base_fading:
            avg_bf = np.mean(self.episode_base_fading)
            std_bf = np.std(self.episode_base_fading)
            self.logger.record("custom/avg_base_fading_db", avg_bf)
            self.logger.record("custom/std_base_fading_db", std_bf)

        # ── fading: user small-scale ──────────────────────────────────────────
        avg_uf = std_uf = 0.0
        if self.episode_user_fading:
            avg_uf = np.mean(self.episode_user_fading)
            std_uf = np.std(self.episode_user_fading)
            self.logger.record("custom/avg_user_fading_db", avg_uf)
            self.logger.record("custom/std_user_fading_db", std_uf)

        # ── optional console print ────────────────────────────────────────────
        if self.verbose > 0:
            print(f"\n=== Episode {self.episodes_finished + 1} Summary (RecurrentPPO) ===")
            print(f"  Avg Data Rate       : {avg_rate:.4f} bits/Hz")
            print(f"  Avg Data Rate       : {avg_rate_kbps:.2f} kbps")
            print(f"  Total UAV Energy    : {total_uav_energy_kj:.4f} kJ")
            print(f"  Total GU Energy     : {total_gu_energy_kj:.6f} kJ")
            print(f"  Total Weighted Eng  : {total_weighted_energy:.4f}")
            print(f"  QoS Satisfaction    : {qos_satisfaction_rate:.1f}%")
            print(f"  Energy Efficiency   : {energy_efficiency:.4f}")
            if self.episode_base_shadowing:
                print(f"  Base Shadowing      : {avg_bs:.2f} +/- {std_bs:.2f} dB")
                print(f"  User Shadowing      : {avg_us:.2f} +/- {std_us:.2f} dB")
                print(f"  Base Fading         : {avg_bf:.2f} +/- {std_bf:.2f} dB")
                print(f"  User Fading         : {avg_uf:.2f} +/- {std_uf:.2f} dB")
            print(f"  Episode Steps       : {self.episode_steps}")
            print("=" * 48 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    def _reset_episode_buffers(self):
        """Clear all accumulators and increment episode counter."""
        self.episode_data_rates        = []
        self.episode_data_rates_kbps   = []
        self.episode_uav_energies      = []
        self.episode_gu_energies       = []
        self.episode_weighted_energies = []
        self.episode_qos_met           = []
        self.episode_base_shadowing    = []
        self.episode_user_shadowing    = []
        self.episode_base_fading       = []
        self.episode_user_fading       = []
        self.episode_steps             = 0
        self.episodes_finished        += 1
