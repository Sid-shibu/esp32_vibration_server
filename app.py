from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Output file will be named with current date
def get_filename():
    date_str = datetime.now().strftime("_%Y%m%d_%H%M%S")
    return f"esp32_vibration_log{date_str}.csv"

# Ensure CSV file exists with header
filename = get_filename()
if not os.path.exists(filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'acc1', 'acc2', 'acc3'])

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    try:
        timestamp = data.get("timestamp", 0)
        accels = data.get("accels", {})
        acc1 = accels.get("acc1", 0)
        acc2 = accels.get("acc2", 0)
        acc3 = accels.get("acc3", 0)

        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, acc1, acc2, acc3])

        print(f"✅ Received: {timestamp}, {acc1}, {acc2}, {acc3}")
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Failed to parse data"}), 500

if __name__ == '__main__':
    print(f"✅ Saving to: {filename}")
    app.run(host='0.0.0.0', port=5000)