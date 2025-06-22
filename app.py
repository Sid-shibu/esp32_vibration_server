from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

# 📁 Prompt for filename once
user_input = input("📁 Enter desired CSV filename (no extension): ").strip()
if not user_input:
    user_input = "esp32_data"
filename = f"{user_input}.csv"
filepath = os.path.join(os.getcwd(), filename)

# 📌 Write CSV header if not already present
if not os.path.exists(filepath):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'acc1_x', 'acc1_y', 'acc1_z',
            'acc2_x', 'acc2_y', 'acc2_z',
            'acc3_x', 'acc3_y', 'acc3_z'
        ])

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    try:
        timestamps = data.get("timestamps", [])
        accels = data.get("accels", {})
        acc1 = accels.get("acc1", [])
        acc2 = accels.get("acc2", [])
        acc3 = accels.get("acc3", [])

        # Check lengths match
        if not (len(timestamps) == len(acc1) == len(acc2) == len(acc3)):
            return jsonify({"error": "Mismatched batch sizes"}), 400

        # Write each row: timestamp, acc1[x,y,z], acc2[x,y,z], acc3[x,y,z]
        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(timestamps)):
                row = [
                    timestamps[i],
                    *acc1[i],  # x, y, z
                    *acc2[i],
                    *acc3[i]
                ]
                writer.writerow(row)

        print(f"✅ Received and saved {len(timestamps)} samples")
        return jsonify({"status": "batch received"}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"✅ Flask server running — saving to: {filepath}")
    app.run(host='0.0.0.0', port=5000)
