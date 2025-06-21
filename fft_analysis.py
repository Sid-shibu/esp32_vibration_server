import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import fft, fftfreq
from numpy import hanning

# === User Input (use in terminal/IDE, not notebook) ===
input_file = input("Enter the CSV filename (e.g., vibration_log.csv): ").strip()

if not os.path.exists(input_file):
    print(f"❌ File not found: {input_file}")
    exit()

# === Constants ===
OUTPUT_DIR = f"fft_dual_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACCEL_SCALE = 9.81 / 16384.0
AMPLIFY_FACTOR = 10.0
MASS = 0.05
ARM_LENGTH = 0.1
SAMPLES = 256
SAMPLING_FREQ = 250

df = pd.read_csv(input_file)
summary = []

# === Analyze Each Sensor ===
for sensor in ['acc1', 'acc2', 'acc3']:
    if sensor not in df.columns:
        print(f"⚠️ Skipping {sensor} (not found in file)")
        continue

    raw = df[sensor].values[:SAMPLES]
    signal = raw - np.mean(raw)
    accel_raw = signal * ACCEL_SCALE
    accel_amplified = accel_raw * AMPLIFY_FACTOR

    def compute_fft(accel):
        windowed = accel * hanning(SAMPLES)
        fft_result = np.abs(fft(windowed))[:SAMPLES//2]
        freqs = fftfreq(SAMPLES, 1/SAMPLING_FREQ)[:SAMPLES//2]
        forces = MASS * accel
        torques = forces * ARM_LENGTH
        peak = np.argmax(fft_result)
        return {
            "freqs": freqs,
            "fft": fft_result,
            "accel": accel[:SAMPLES//2],
            "force": forces[:SAMPLES//2],
            "torque": torques[:SAMPLES//2],
            "peak_freq": freqs[peak],
            "peak_accel": abs(accel[peak]),
            "avg_accel": np.mean(np.abs(accel)),
            "peak_torque": abs(torques[peak]),
            "avg_torque": np.mean(np.abs(torques))
        }

    raw_data = compute_fft(accel_raw)
    amp_data = compute_fft(accel_amplified)

    # Save raw + amp to CSV
    pd.DataFrame({
        'bin': range(1, SAMPLES//2 + 1),
        'frequency': raw_data['freqs'],
        'magnitude_raw': raw_data['fft'],
        'acceleration_raw': raw_data['accel'],
        'force_raw': raw_data['force'],
        'torque_raw': raw_data['torque']
    }).to_csv(os.path.join(OUTPUT_DIR, f"{sensor}_raw.csv"), index=False)

    pd.DataFrame({
        'bin': range(1, SAMPLES//2 + 1),
        'frequency': amp_data['freqs'],
        'magnitude_amplified': amp_data['fft'],
        'acceleration_amplified': amp_data['accel'],
        'force_amplified': amp_data['force'],
        'torque_amplified': amp_data['torque']
    }).to_csv(os.path.join(OUTPUT_DIR, f"{sensor}_amplified.csv"), index=False)

    # Plot both FFTs
    plt.figure(figsize=(10, 5))
    plt.plot(raw_data['freqs'], raw_data['fft'], label="Raw FFT")
    plt.plot(amp_data['freqs'], amp_data['fft'], '--', label="Amplified FFT ×10")
    plt.title(f"FFT Comparison - {sensor}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{sensor}_fft_comparison.png"))
    plt.close()

    # Summary
    summary.append({
        'sensor': sensor,
        'raw_peak_freq': raw_data['peak_freq'],
        'raw_peak_accel': raw_data['peak_accel'],
        'raw_avg_accel': raw_data['avg_accel'],
        'raw_peak_torque': raw_data['peak_torque'],
        'raw_avg_torque': raw_data['avg_torque'],
        'amp_peak_freq': amp_data['peak_freq'],
        'amp_peak_accel': amp_data['peak_accel'],
        'amp_avg_accel': amp_data['avg_accel'],
        'amp_peak_torque': amp_data['peak_torque'],
        'amp_avg_torque': amp_data['avg_torque']
    })

# Save Summary
summary_df = pd.DataFrame(summary)
summary_df.to_excel(os.path.join(OUTPUT_DIR, "summary_stats.xlsx"), index=False)

print(f"✅ Done! All results saved in: {OUTPUT_DIR}")
