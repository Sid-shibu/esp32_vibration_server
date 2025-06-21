# ✅ Flask Server to Receive FFT from ESP32 and Save All Data in One CSV and Excel
from flask import Flask, request, jsonify
import os, csv
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # ✅ Prevent Tkinter GUI errors
import matplotlib.pyplot as plt

app = Flask(__name__)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"fft_data_{timestamp}"
os.makedirs(base_dir, exist_ok=True)

csv_path = os.path.join(base_dir, "all_sensors_fft.csv")
excel_path = os.path.join(base_dir, "all_sensors_fft.xlsx")
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Create CSV file with headers
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["sensor", "bin", "magnitude"])

@app.route('/upload', methods=['POST'])
def receive_fft():
    data = request.get_json()
    if not data or "sensor" not in data or "fft" not in data:
        return jsonify({"error": "Invalid JSON"}), 400

    sensor = data["sensor"]
    fft_values = data["fft"]

    # Append to single CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, val in enumerate(fft_values, start=1):
            writer.writerow([sensor, i, val])

    # Plot and save as image
    plt.figure()
    plt.plot(range(1, len(fft_values) + 1), fft_values)
    plt.title(f"FFT - {sensor}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{sensor}.png"))
    plt.close()

    print(f"✅ FFT data received from: {sensor}")
    return jsonify({"status": "ok", "sensor": sensor})

@app.route('/export', methods=['GET'])
def export_excel():
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(csv_path)
    df.to_excel(excel_path, index=False)

    return jsonify({"status": "exported", "file": excel_path}), 200

if __name__ == '__main__':
    print(f"🚀 FFT Server started. Saving to: {base_dir}")
    app.run(host='0.0.0.0', port=5000)
