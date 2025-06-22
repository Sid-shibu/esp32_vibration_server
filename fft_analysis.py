import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import fft, fftfreq
from numpy import hanning

# === Configuration ===
ACCEL_SCALE = 9.81 / 16384.0  # Scaling raw data to m/s^2
AMPLIFY_FACTOR = 10.0         # Amplify weak tremor signals
MASS = 0.05                   # Assumed finger segment mass in kg
ARM_LENGTH = 0.1              # Lever arm in meters
TREMOR_FREQ_RANGE = (4, 12)   # Tremor frequency band (Hz)
TREMOR_MAG_THRESHOLD = 0.05   # Lowered threshold for normalized FFT

# === File Input ===
filename = input("📂 Enter the name of your CSV file (e.g., Shahul.csv): ").strip()
if not os.path.exists(filename):
    raise FileNotFoundError(f"❌ File not found: {filename}")

df = pd.read_csv(filename)

# === Dynamically Calculate Sampling Frequency ===
if 'timestamp' in df.columns:
    timestamps = df['timestamp'].dropna().values
    time_diffs = np.diff(timestamps)
    mean_diff_sec = np.mean(time_diffs) / 1e6  # assuming microseconds
    SAMPLING_FREQ = 1 / mean_diff_sec
else:
    raise KeyError("Column 'timestamp' not found for dynamic sampling rate calculation.")

# === Prepare Output Folder ===
OUTPUT_DIR = f"fft_tremor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = []

# === Analyze Each Sensor ===
for sensor in [col for col in df.columns if col.startswith('acc')]:
    raw_full = df[sensor].dropna().values
    SAMPLES = min(len(raw_full), 1024)  # Use up to 1024 samples (≈4 sec at 250 Hz)
    if SAMPLES < 64:
        continue

    raw = raw_full[:SAMPLES]
    signal = raw - np.mean(raw)
    accel_raw = signal * ACCEL_SCALE
    accel_amplified = accel_raw * AMPLIFY_FACTOR

    def compute_fft(accel):
        windowed = accel * hanning(len(accel))
        fft_result = np.abs(fft(windowed))[:len(accel)//2]
        fft_result /= len(accel)  # Normalize
        freqs = fftfreq(len(accel), 1/SAMPLING_FREQ)[:len(accel)//2]
        forces = MASS * accel
        torques = forces * ARM_LENGTH
        peak = np.argmax(fft_result)
        avg_accel = np.mean(np.abs(accel))
        return {
            "freqs": freqs,
            "fft": fft_result,
            "accel": accel[:len(accel)//2],
            "force": forces[:len(accel)//2],
            "torque": torques[:len(accel)//2],
            "peak_freq": freqs[peak],
            "avg_freq": np.average(freqs, weights=fft_result),
            "peak_accel": abs(accel[peak]),
            "avg_accel": avg_accel,
            "peak_torque": abs(torques[peak]),
            "avg_torque": np.mean(np.abs(torques))
        }

    raw_data = compute_fft(accel_raw)
    amp_data = compute_fft(accel_amplified)

    # Tremor Detection
    tremor_band = (amp_data["freqs"] >= TREMOR_FREQ_RANGE[0]) & (amp_data["freqs"] <= TREMOR_FREQ_RANGE[1])
    tremor_magnitude = amp_data["fft"][tremor_band]
    tremor_peak_mag = np.max(tremor_magnitude) if len(tremor_magnitude) > 0 else 0
    tremor_detected = tremor_peak_mag > TREMOR_MAG_THRESHOLD

    # Save FFT CSVs
    pd.DataFrame({
        'bin': range(1, len(raw_data['freqs']) + 1),
        'frequency': raw_data['freqs'],
        'magnitude_raw': raw_data['fft'],
        'acceleration_raw': raw_data['accel'],
        'force_raw': raw_data['force'],
        'torque_raw': raw_data['torque']
    }).to_csv(os.path.join(OUTPUT_DIR, f"{sensor}_raw.csv"), index=False)

    pd.DataFrame({
        'bin': range(1, len(amp_data['freqs']) + 1),
        'frequency': amp_data['freqs'],
        'magnitude_amplified': amp_data['fft'],
        'acceleration_amplified': amp_data['accel'],
        'force_amplified': amp_data['force'],
        'torque_amplified': amp_data['torque']
    }).to_csv(os.path.join(OUTPUT_DIR, f"{sensor}_amplified.csv"), index=False)

    # Plot FFT Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(raw_data['freqs'], raw_data['fft'], label="Raw FFT")
    plt.plot(amp_data['freqs'], amp_data['fft'], '--', label="Amplified FFT ×10")
    plt.axvspan(*TREMOR_FREQ_RANGE, color='red', alpha=0.1, label="Tremor Band (4–12 Hz)")
    plt.title(f"FFT Comparison - {sensor}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Magnitude (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{sensor}_fft_comparison.png"))
    plt.close()

    # Summary Data
    summary.append({
        'sensor': sensor,
        'sampling_rate_Hz': round(SAMPLING_FREQ, 2),
        'samples_used': SAMPLES,
        'raw_peak_freq_Hz': raw_data['peak_freq'],
        'raw_avg_freq_Hz': raw_data['avg_freq'],
        'raw_peak_accel_m/s2': raw_data['peak_accel'],
        'raw_avg_accel_m/s2': raw_data['avg_accel'],
        'raw_peak_torque_Nm': raw_data['peak_torque'],
        'raw_avg_torque_Nm': raw_data['avg_torque'],
        'amp_peak_freq_Hz': amp_data['peak_freq'],
        'amp_avg_freq_Hz': amp_data['avg_freq'],
        'amp_peak_accel_m/s2': amp_data['peak_accel'],
        'amp_avg_accel_m/s2': amp_data['avg_accel'],
        'amp_peak_torque_Nm': amp_data['peak_torque'],
        'amp_avg_torque_Nm': amp_data['avg_torque'],
        'tremor_peak_mag': tremor_peak_mag,
        'tremor_detected': tremor_detected
    })

# Save Summary
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTPUT_DIR, "summary_stats_with_tremor.xlsx")
summary_df.to_excel(summary_path, index=False)

print(f"\n✅ Output directory: {OUTPUT_DIR}")
print(f"📊 Summary file saved at: {summary_path}")
