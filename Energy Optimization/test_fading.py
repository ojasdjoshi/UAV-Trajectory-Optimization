"""
Test script to validate fading implementation and visualize channel characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from uavenv import UavEnv, FadingChannel, Entity
import os

# Create output directory
os.makedirs("fading_tests", exist_ok=True)

def test_rician_fading_distribution():
    """Test that Rician fading follows expected distribution."""
    print("\n" + "="*60)
    print("Test 1: Rician Fading Distribution")
    print("="*60)
    
    channel = FadingChannel("test")
    n_samples = 10000
    power_gains = []
    
    for _ in range(n_samples):
        power_gain = channel.generate_small_scale_fading(is_los=True)
        power_gains.append(power_gain)
    
    power_gains_db = 10 * np.log10(power_gains)
    
    print(f"Mean power gain: {np.mean(power_gains):.4f}")
    print(f"Std dev power gain: {np.std(power_gains):.4f}")
    print(f"Mean (dB): {np.mean(power_gains_db):.2f} dB")
    print(f"Std dev (dB): {np.std(power_gains_db):.2f} dB")
    
    # Plot histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(power_gains, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Power Gain')
    plt.ylabel('Probability Density')
    plt.title('Rician Fading - Power Gain Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(power_gains_db, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Power Gain (dB)')
    plt.ylabel('Probability Density')
    plt.title('Rician Fading - Power Gain (dB)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fading_tests/rician_distribution.png', dpi=150)
    print("Saved: fading_tests/rician_distribution.png")
    plt.close()


def test_rayleigh_fading_distribution():
    """Test that Rayleigh fading follows expected distribution."""
    print("\n" + "="*60)
    print("Test 2: Rayleigh Fading Distribution")
    print("="*60)
    
    channel = FadingChannel("test")
    n_samples = 10000
    power_gains = []
    
    for _ in range(n_samples):
        power_gain = channel.generate_small_scale_fading(is_los=False)
        power_gains.append(power_gain)
    
    power_gains_db = 10 * np.log10(power_gains)
    
    print(f"Mean power gain: {np.mean(power_gains):.4f}")
    print(f"Std dev power gain: {np.std(power_gains):.4f}")
    print(f"Mean (dB): {np.mean(power_gains_db):.2f} dB")
    print(f"Std dev (dB): {np.std(power_gains_db):.2f} dB")
    print(f"Expected mean (Rayleigh): 1.0")
    
    # Plot histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(power_gains, bins=50, density=True, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('Power Gain')
    plt.ylabel('Probability Density')
    plt.title('Rayleigh Fading - Power Gain Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(power_gains_db, bins=50, density=True, alpha=0.7, edgecolor='black', color='red')
    plt.xlabel('Power Gain (dB)')
    plt.ylabel('Probability Density')
    plt.title('Rayleigh Fading - Power Gain (dB)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fading_tests/rayleigh_distribution.png', dpi=150)
    print("Saved: fading_tests/rayleigh_distribution.png")
    plt.close()


def test_shadowing_distribution():
    """Test that shadowing follows log-normal distribution."""
    print("\n" + "="*60)
    print("Test 3: Shadowing Distribution")
    print("="*60)
    
    channel = FadingChannel("test")
    n_samples = 1000
    shadowing_values = []
    
    for i in range(n_samples):
        # Force update each time to get independent samples
        pos = Entity(x=i*100, y=0, z=50)  # Spread out positions
        channel.update_shadowing(pos, is_los=True, force_update=True)
        shadowing_values.append(channel.shadowing_db)
    
    print(f"Mean shadowing: {np.mean(shadowing_values):.2f} dB")
    print(f"Std dev shadowing: {np.std(shadowing_values):.2f} dB")
    print(f"Expected std dev: 8.0 dB")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(shadowing_values, bins=50, density=True, alpha=0.7, edgecolor='black', color='purple')
    plt.xlabel('Shadowing (dB)')
    plt.ylabel('Probability Density')
    plt.title('Log-Normal Shadowing Distribution')
    plt.grid(True, alpha=0.3)
    
    # Overlay theoretical normal distribution
    x = np.linspace(min(shadowing_values), max(shadowing_values), 100)
    theoretical = (1 / (8.0 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / 8.0)**2)
    plt.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical N(0, 8²)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fading_tests/shadowing_distribution.png', dpi=150)
    print("Saved: fading_tests/shadowing_distribution.png")
    plt.close()


def test_shadowing_spatial_correlation():
    """Test spatial correlation of shadowing."""
    print("\n" + "="*60)
    print("Test 4: Shadowing Spatial Correlation")
    print("="*60)
    
    channel = FadingChannel("test")
    
    # Simulate UAV moving in a straight line
    positions = []
    shadowing_values = []
    
    pos = Entity(x=0, y=250, z=75)
    channel.update_shadowing(pos, is_los=True, force_update=True)
    
    for i in range(200):
        pos.prev_x = pos.x
        pos.prev_y = pos.y
        pos.x += 2  # Move 2 meters per step
        
        channel.update_shadowing(pos, is_los=True)
        positions.append(pos.x)
        shadowing_values.append(channel.shadowing_db)
    
    print(f"Total distance traveled: {positions[-1] - positions[0]} meters")
    print(f"Shadowing range: [{min(shadowing_values):.2f}, {max(shadowing_values):.2f}] dB")
    
    # Plot shadowing vs position
    plt.figure(figsize=(12, 6))
    plt.plot(positions, shadowing_values, 'b-', linewidth=1.5)
    plt.xlabel('Position (meters)')
    plt.ylabel('Shadowing (dB)')
    plt.title('Shadowing Spatial Correlation (Moving 2m per step)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean (0 dB)')
    plt.axhline(y=8, color='orange', linestyle='--', alpha=0.5, label='+1 std dev')
    plt.axhline(y=-8, color='orange', linestyle='--', alpha=0.5, label='-1 std dev')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fading_tests/shadowing_correlation.png', dpi=150)
    print("Saved: fading_tests/shadowing_correlation.png")
    plt.close()


def test_channel_variation_in_environment():
    """Test channel variation in actual environment."""
    print("\n" + "="*60)
    print("Test 5: Channel Variation in Environment")
    print("="*60)
    
    env = UavEnv()
    obs, info = env.reset()
    
    data_rates = []
    base_shadowing = []
    user_shadowing = []
    base_fading = []
    user_fading = []
    
    # Run for 100 steps
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        data_rates.append(info['data_rate'])
        if 'fading' in info:
            base_shadowing.append(info['fading']['base_shadowing'])
            user_shadowing.append(info['fading']['user_shadowing'])
            base_fading.append(info['fading']['base_fading'])
            user_fading.append(info['fading']['user_fading'])
        
        if done:
            break
    
    print(f"Steps completed: {len(data_rates)}")
    print(f"Average data rate: {np.mean(data_rates):.4f} bits/Hz")
    print(f"Data rate std dev: {np.std(data_rates):.4f} bits/Hz")
    print(f"Average base shadowing: {np.mean(base_shadowing):.2f} dB")
    print(f"Average user shadowing: {np.mean(user_shadowing):.2f} dB")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Data rate
    axes[0].plot(data_rates, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Data Rate (bits/Hz)')
    axes[0].set_title('Channel Quality Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(data_rates), color='r', linestyle='--', alpha=0.5, label='Mean')
    axes[0].legend()
    
    # Shadowing
    axes[1].plot(base_shadowing, 'g-', linewidth=1.5, label='Base Link')
    axes[1].plot(user_shadowing, 'orange', linewidth=1.5, label='User Link')
    axes[1].set_ylabel('Shadowing (dB)')
    axes[1].set_title('Shadowing Variation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Small-scale fading
    axes[2].plot(base_fading, 'purple', linewidth=1.5, alpha=0.7, label='Base Link')
    axes[2].plot(user_fading, 'red', linewidth=1.5, alpha=0.7, label='User Link')
    axes[2].set_ylabel('Fading Gain (dB)')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Small-Scale Fading Variation')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fading_tests/environment_channel_variation.png', dpi=150)
    print("Saved: fading_tests/environment_channel_variation.png")
    plt.close()
    
    env.close()


def run_all_tests():
    """Run all fading tests."""
    print("\n" + "="*70)
    print(" FADING IMPLEMENTATION VALIDATION TESTS")
    print("="*70)
    
    test_rician_fading_distribution()
    test_rayleigh_fading_distribution()
    test_shadowing_distribution()
    test_shadowing_spatial_correlation()
    test_channel_variation_in_environment()
    
    print("\n" + "="*70)
    print(" ALL TESTS COMPLETED")
    print("="*70)
    print(f"\nAll plots saved to 'fading_tests/' directory")
    print("\nExpected Results:")
    print("  - Rician fading: Mean power gain > 1 (due to LOS component)")
    print("  - Rayleigh fading: Mean power gain ≈ 1")
    print("  - Shadowing: Mean ≈ 0 dB, Std dev ≈ 8 dB")
    print("  - Spatial correlation: Smooth variation over distance")
    print("  - Channel quality: Varies over time due to fading")
    

if __name__ == "__main__":
    run_all_tests()
