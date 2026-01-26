# client_test.py
import requests
import json
import random

# Server address (localhost if running on the same machine)
URL = "http://127.0.0.1:8000/predict_pairs"

# Data simulation: [ [Pair_ID, [5 RSSI values]], ... ]
# Example: Data from 3 different beacons
data_to_send = [
    ["beacon_A1", [-60, -62, -61, -59, -60]],
    ["beacon_B2", [-85, -84, -86, -88, -85]],
    ["beacon_C3", [-45, -44, -46, -45, -44]]
]

try:
    print(f"📡 Sending {len(data_to_send)} packets to the server...")

    # Sending POST request
    response = requests.post(URL, json=data_to_send)

    # Checking status
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Response received from server:")
        print("-" * 30)
        print(f"{'PAIR ID':<15} | {'DISTANCE (m)':<10}")
        print("-" * 30)

        for key, distance in result:
            print(f"{key:<15} | {distance:.2f} m")
        print("-" * 30)
    else:
        print(f"❌ Server error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ Failed to connect to server. Ensure 'server_pairs.py' is running.")