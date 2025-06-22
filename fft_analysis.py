import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ========================
# MEDICAL ANALYSIS CONFIG
# ========================
TREMOR_BANDS = {
    'Parkinsonian': (4, 6),
    'Essential': (6, 12),
    'Physiological': (8, 12),
    'Cerebellar': (3, 8)
}

UPDRS_SCORING = {
    'amplitude': [(0.1, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 10.0)],
    'frequency': [(3, 4), (4, 6), (6, 8), (8, 12)]
}

# ========================
# SUPPRESSION SYSTEM DESIGN
# ========================
SUPPRESSION_PARAMS = {
    'required_bandwidth': 20,  # Hz
    'max_actuator_force': 5.0,  # Newtons
    'response_latency': 0.05   # seconds
}

def clinical_tremor_analysis(data, fs):
    """Comprehensive tremor analysis for medical diagnosis"""
    results = {}
    
    # 1. Time Domain Analysis
    results['time_features'] = {
        'rms': np.sqrt(np.mean(data**2)),
        'peak_to_peak': np.ptp(data),
        'zero_crossings': len(np.where(np.diff(np.sign(data)))[0])
    }
    
    # 2. Frequency Domain Analysis
    n = len(data)
    yf = fftpack.fft(data * np.hanning(n))
    xf = fftpack.fftfreq(n, 1/fs)[:n//2]
    yf_abs = 2.0/n * np.abs(yf[:n//2])
    
    # Find dominant tremor components
    peaks, _ = signal.find_peaks(yf_abs, height=0.1*np.max(yf_abs))
    dominant_freqs = xf[peaks]
    dominant_amps = yf_abs[peaks]
    
    # 3. Tremor Classification
    tremor_type = "Undetermined"
    for name, band in TREMOR_BANDS.items():
        if any((band[0] <= f <= band[1]) for f in dominant_freqs):
            tremor_type = name
            break
    
    # 4. UPDRS Scoring
    updrs_score = 0
    max_amp = np.max(dominant_amps)
    for i, (low, high) in enumerate(UPDRS_SCORING['amplitude']):
        if low <= max_amp < high:
            updrs_score = i + 1
            break
    
    # 5. Suppression System Parameters
    suppression_params = {
        'target_frequencies': dominant_freqs.tolist(),
        'required_bandwidth': max(dominant_freqs) - min(dominant_freqs),
        'actuator_force': 0.2 * max_amp * SUPPRESSION_PARAMS['max_actuator_force']
    }
    
    return {
        'time_features': results['time_features'],
        'frequency_peaks': list(zip(dominant_freqs, dominant_amps)),
        'tremor_type': tremor_type,
        'updrs_score': updrs_score,
        'suppression_params': suppression_params,
        'raw_fft': (xf, yf_abs)
    }

def plot_3d_trajectory(ax, x, y, z, title=""):
    """Plot 3D trajectory of tremor"""
    ax.plot(x, y, z, alpha=0.6)
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='x', label='End')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    ax.legend()

def generate_medical_report(results, output_dir):
    """Generate comprehensive medical report"""
    report = f"""
    ======================
    TREMOR ANALYSIS REPORT
    ======================
    
    Patient ID: {results.get('patient_id', 'N/A')}
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    
    [CLINICAL FINDINGS]
    - Dominant Tremor Type: {results['tremor_type']}
    - UPDRS Severity Score: {results['updrs_score']}/4
    - Peak Tremor Frequency: {results['frequency_peaks'][0][0]:.2f} Hz
    - Maximum Amplitude: {results['frequency_peaks'][0][1]:.4f} m/s²
    
    [SUPPRESSION SYSTEM PARAMETERS]
    - Target Frequencies: {', '.join(f'{f:.2f}Hz' for f in results['suppression_params']['target_frequencies'])}
    - Required Bandwidth: {results['suppression_params']['required_bandwidth']:.2f} Hz
    - Recommended Actuator Force: {results['suppression_params']['actuator_force']:.2f} N
    
    [RECOMMENDATIONS]
    {get_clinical_recommendations(results)}
    """
    
    with open(os.path.join(output_dir, 'medical_report.txt'), 'w') as f:
        f.write(report)
    
    return report

def get_clinical_recommendations(results):
    """Generate clinical recommendations based on analysis"""
    if results['updrs_score'] >= 3:
        return "Consider pharmacological intervention combined with mechanical suppression"
    elif results['updrs_score'] == 2:
        return "Mechanical suppression recommended with optional medication"
    else:
        return "Monitor progression; consider lifestyle modifications"

# ========================
# MAIN PROCESSING PIPELINE
# ========================
def analyze_tremor_data(filename):
    # Load data
    df = pd.read_csv(filename)
    
    # Create output directory
    output_dir = f"tremor_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sensor
    sensor_results = {}
    for sensor in ['thumb', 'index', 'middle']:
        try:
            x = df[f'{sensor}_x'].values
            y = df[f'{sensor}_y'].values
            z = df[f'{sensor}_z'].values
            
            # Calculate magnitude
            mag = np.sqrt(x**2 + y**2 + z**2)
            
            # Perform analysis
            fs = 250  # Sampling frequency (Hz)
            results = clinical_tremor_analysis(mag, fs)
            
            # Store results
            sensor_results[sensor] = results
            
            # Generate plots
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(mag)
            plt.title(f'{sensor} - Time Domain')
            
            plt.subplot(2, 1, 2)
            plt.plot(*results['raw_fft'])
            plt.title(f'{sensor} - Frequency Domain')
            plt.savefig(os.path.join(output_dir, f'{sensor}_analysis.png'))
            plt.close()
            
            # 3D Trajectory Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            plot_3d_trajectory(ax, x, y, z, f'{sensor} 3D Tremor Path')
            plt.savefig(os.path.join(output_dir, f'{sensor}_3d_path.png'))
            plt.close()
            
        except KeyError:
            print(f"Skipping {sensor} - missing data columns")
    
    # Generate comprehensive report
    main_results = {
        'patient_id': input("Enter patient ID: "),
        **sensor_results
    }
    report = generate_medical_report(main_results, output_dir)
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print(report)
    
    return main_results

# Run analysis
if __name__ == "__main__":
    filename = input("Enter data filename: ")
    results = analyze_tremor_data(filename)
