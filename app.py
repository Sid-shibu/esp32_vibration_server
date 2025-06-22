from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

# 📝 Ask user for filename at startup
user_input = input("📁 Enter the desired CSV filename (without .csv): ").strip()
if not user_input:
    user_input = "esp32_data"
filename = f"{user_input}.csv"

# ✅ Ensure file is in current directory and writable
filepath = os.path.join(os.getcwd(), filename)

# 🧱 Create file with header if it doesn't exist
if not os.path.exists(filepath):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'acc1', 'acc2', 'acc3'])

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()

    # 🔍 Debug: Print incoming JSON
    print("🔄 Incoming JSON:", data)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    try:
        timestamps = data.get("timestamps", [])
        accels = data.get("accels", {})
        acc1_list = accels.get("acc1", [])
        acc2_list = accels.get("acc2", [])
        acc3_list = accels.get("acc3", [])

        rows = zip(timestamps, acc1_list, acc2_list, acc3_list)

        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        print(f"✅ Received batch of {len(timestamps)} samples")
        return jsonify({"status": "batch received"}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Failed to parse batch"}), 500

if __name__ == '__main__':
    print(f"✅ Flask server started. Saving to: {filepath}")
    app.run(host='0.0.0.0', port=5000)
