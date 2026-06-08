import subprocess
import threading
import time
import os
import csv
import signal
import math
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INTERFACE = "wlan0"
FEATURES_DIR = "/home/shiyiz/iot_project/features"
AUTO_SAVE_INTERVAL = 30

# Ports to monitor
MONITORED_PORTS = [21, 22, 23, 80, 443, 2244, 3000, 5000, 8080, 8086]

# Web ports — save flow immediately on FIN/RST (per-request granularity)
WEB_PORTS = {80, 443, 5000, 8080, 8086, 3000}

PORT_LABELS = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    80: "HTTP",
    443: "HTTPS",
    2244: "SSH-Custom",
    3000: "Grafana",
    5000: "Flask/IoT",
    8080: "HTTP-Alt",
    8086: "InfluxDB",
}

# Protocol detection by port
PORT_PROTOCOL = {
    21: 'FTP', 22: 'SSH', 23: 'Telnet', 80: 'HTTP', 443: 'HTTPS',
    2244: 'SSH', 3000: 'HTTP', 5000: 'HTTP', 8080: 'HTTP', 8086: 'HTTP',
}

PI_IP = "192.168.1.215"
port_filter = " or ".join(["port " + str(p) for p in MONITORED_PORTS])

# ─────────────────────────────────────────────
# Mode Flags
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Mode selection via command-line argument
# Usage: sudo python3 network_extractor.py [mode]
#
# Modes:
#   (none)     normal — ports 21/22/23/80/443/2244/3000/5000/8080/8086
#   ddos       DDoS attacks (random source IPs)
#   dos        DoS TCP floods
#   gre        GRE tunnel attacks
#   recon      Reconnaissance / port scans (all ports)
#   backdoor   Reverse shell / C2 traffic (any port, IP-based)
#   mitm       MitM attacks on sensor port 5000
#
# Example:
#   sudo python3 network_extractor.py mitm
#   sudo python3 network_extractor.py backdoor
#   sudo python3 network_extractor.py ddos
# ─────────────────────────────────────────────
import sys
_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "normal"

GRE_MODE      = (_mode == "gre")
RECON_MODE    = (_mode == "recon")
BACKDOOR_MODE = (_mode == "backdoor")
MITM_MODE     = (_mode == "mitm")
DOS_MODE      = (_mode == "dos")
DDOS_MODE_ARG = (_mode == "ddos")   # resolved below after DDOS_MODE definition

# Backdoor C2 IPs — any TCP between Pi and these IPs = Backdoor, port-agnostic.
# VM1 (192.168.1.151): backdoor installer / reverse shell handler (Meterpreter etc.)
# VM2 (192.168.1.156): data receiver / exfiltration target (nc listener etc.)
# Works for Meterpreter, netcat, Python shells, Cobalt Strike, any future C2.
BACKDOOR_C2_IPS = {"192.168.1.151", "192.168.1.156", "192.168.1.170"}

# MitM VM roles — one VM per attack type, label by src_ip post-capture
# VM1 (192.168.1.150): MitM_ARP        — ARP spoof + data manipulation
# VM2 (192.168.1.156): Spoofing_Sensor — direct sensor impersonation
# VM3 (192.168.1.170): Replay_Sensor   — tcpreplay of captured sensor packets
MITM_VM_IPS = {
    "192.168.1.150": "MitM_ARP",
    "192.168.1.156": "Spoofing_Sensor",
    "192.168.1.170": "Replay_Sensor",
}

# Real sensor IPs — traffic from these in MITM_MODE = intercepted (MitM_ARP)
SENSOR_IPS = {"192.168.1.116", "192.168.1.128", "192.168.1.162"}

if GRE_MODE or RECON_MODE or DDOS_MODE_ARG or DOS_MODE:
    # Capture ALL traffic to RPi - disconnect SSH first!
    # DDoS/DoS need all traffic since src_ip is random/spoofed
    CAPTURE_FILTER = "dst host " + PI_IP
elif BACKDOOR_MODE:
    # Port-agnostic: capture ALL TCP between Pi and any C2 VM (both directions)
    c2_host_filter = " or ".join(["host " + ip for ip in sorted(BACKDOOR_C2_IPS)])
    CAPTURE_FILTER = "tcp and host " + PI_IP + " and (" + c2_host_filter + ")"
elif MITM_MODE:
    # Capture ALL TCP to Pi:5000 — filter out handshake packets in parse_tcpdump_line
    CAPTURE_FILTER = "tcp and dst host " + PI_IP + " and dst port 5000"
else:
    CAPTURE_FILTER = "(((tcp or udp) and (" + port_filter + ")) or icmp or arp) and dst host " + PI_IP

# ─────────────────────────────────────────────
# DDoS Mode - set True when running DDoS attacks
# Groups all traffic by dst_ip+dst_port+time_window
# ignoring src_ip (handles random source IP spoofing)
# ─────────────────────────────────────────────
DDOS_MODE = DDOS_MODE_ARG   # set by command-line arg: sudo python3 extractor.py ddos
DOS_MODE  = (_mode == "dos")
DDOS_ATTACKER_IP = '192.168.1.150'  # VM1 IP for DDoS flows

# UDP/ICMP packet window - group N packets into one flow row
UDP_PACKET_WINDOW = 10   # 10 packets per row for UDP/ICMP

# Known spoofed source IPs - TCP traffic from these IPs uses packet window
SPOOF_IPS = {'192.168.1.100', '192.168.1.101'}

# Memory safety limits
# ─────────────────────────────────────────────
MAX_FLOWS = 5000
MAX_PKT_SIZES_PER_FLOW = 200
FLOW_EXPIRE_SECONDS = 15
EMERGENCY_FLOW_LIMIT = 10000

# ─────────────────────────────────────────────
# InfluxDB
# ─────────────────────────────────────────────
influx_client = None
try:
    from influxdb import InfluxDBClient
    influx_client = InfluxDBClient(
        host='localhost', port=8086,
        username='grafana', password='iot_secure_pass',
        database='iot_sensors'
    )
    influx_client.ping()
    print("✓ InfluxDB connected")
except Exception as e:
    print("⚠ InfluxDB not available: " + str(e))
    print("  Continuing without InfluxDB (CSV only)")
    influx_client = None

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
flow_data = {}
port_stats = {}
device_files = {}
lock = threading.Lock()
stop_capture = False
total_packets = 0
total_flows = 0
dropped_flows = 0
forced_saves = 0

# UDP/ICMP packet buffer: (src_ip, dst_ip, dst_port) -> list of packet dicts
udp_buffers = {}


class FlowInfo:
    __slots__ = [
        'src_ip', 'src_port', 'dst_ip', 'dst_port',
        'start_time', 'end_time',
        'fwd_packets', 'bwd_packets',
        'fwd_bytes', 'bwd_bytes',
        'fwd_pkt_sizes', 'bwd_pkt_sizes',
        'fwd_times', 'bwd_times',
        'fwd_header_bytes', 'bwd_header_bytes',
        'syn', 'fin', 'rst', 'psh', 'ack', 'ece', 'cwr', 'urg',
        'protocol',  # 6=TCP, 17=UDP
    ]

    def __init__(self, src_ip, src_port, dst_ip, dst_port):
        self.src_ip = src_ip
        self.src_port = int(src_port)
        self.dst_ip = dst_ip
        self.dst_port = int(dst_port)
        self.start_time = None
        self.end_time = None
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_pkt_sizes = []
        self.bwd_pkt_sizes = []
        self.fwd_times = []
        self.bwd_times = []
        self.fwd_header_bytes = 0
        self.bwd_header_bytes = 0
        self.syn = 0
        self.fin = 0
        self.rst = 0
        self.psh = 0
        self.ack = 0
        self.ece = 0
        self.cwr = 0
        self.urg = 0
        self.protocol = 6  # Default TCP

    @property
    def service_port(self):
        if self.dst_port in MONITORED_PORTS:
            return self.dst_port
        if self.src_port in MONITORED_PORTS:
            return self.src_port
        return self.dst_port


# ─────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────
def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def variance(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)

def magnitude(fwd_lens, bwd_lens):
    """Sqrt of sum of squares of means"""
    fwd_m = mean(fwd_lens)
    bwd_m = mean(bwd_lens)
    return math.sqrt(fwd_m ** 2 + bwd_m ** 2)

def radius(fwd_lens, bwd_lens):
    """Sqrt of sum of variances"""
    fwd_v = variance(fwd_lens)
    bwd_v = variance(bwd_lens)
    return math.sqrt(fwd_v + bwd_v)

def covariance(fwd_lens, bwd_lens):
    """Covariance between fwd and bwd packet lengths"""
    n = min(len(fwd_lens), len(bwd_lens))
    if n < 2:
        return 0
    fwd_m = mean(fwd_lens[:n])
    bwd_m = mean(bwd_lens[:n])
    cov = sum((fwd_lens[i] - fwd_m) * (bwd_lens[i] - bwd_m) for i in range(n)) / (n - 1)
    return cov


def signal_handler(sig, frame):
    global stop_capture
    stop_capture = True
    print("\n\n🛑 Stopping capture...", flush=True)


# ─────────────────────────────────────────────
# Packet parsing
# ─────────────────────────────────────────────
TCP_HEADER_SIZE = 54  # IP (20) + TCP (20) + Ethernet (14)
UDP_HEADER_SIZE = 42  # IP (20) + UDP (8) + Ethernet (14)
ICMP_HEADER_SIZE = 42 # IP (20) + ICMP (8) + Ethernet (14)
ARP_HEADER_SIZE = 42  # ARP (28) + Ethernet (14)

def flush_udp_buffer(buf_key, device_ip):
    global total_flows
    pkts = udp_buffers.pop(buf_key, [])
    if not pkts:
        return

    src_ip   = pkts[0]['src_ip']
    dst_ip   = pkts[0]['dst_ip']
    dst_port = pkts[0]['dst_port']

    fi = FlowInfo(src_ip, 0, dst_ip, dst_port)
    # Use TCP protocol if src_ip is a known spoof IP, otherwise UDP
    fi.protocol = 6 if (src_ip in SPOOF_IPS or DOS_MODE or DDOS_MODE or RECON_MODE) else 17
    fi.start_time = pkts[0]['timestamp']
    fi.end_time   = pkts[-1]['timestamp']

    for pkt in pkts:
        fi.fwd_packets += 1
        fi.fwd_bytes   += pkt['pkt_len']
        fi.fwd_header_bytes += pkt['header_size']
        if len(fi.fwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
            fi.fwd_pkt_sizes.append(pkt['pkt_len'])
            fi.fwd_times.append(pkt['timestamp'])
        # Count TCP flags if present
        f = pkt.get('flags', '')
        if 'S' in f: fi.syn += 1
        if 'F' in f: fi.fin += 1
        if 'R' in f: fi.rst += 1
        if 'P' in f: fi.psh += 1
        if '.' in f: fi.ack += 1
        if 'E' in f: fi.ece += 1
        if 'W' in f: fi.cwr += 1
        if 'U' in f: fi.urg += 1

    total_flows += 1
    features = extract_features(fi)
    if not features:
        return

    if device_ip not in device_files:
        ts = datetime.now().strftime("%d%m")
        short_ip = device_ip.split('.')[-1]
        filename = os.path.join(FEATURES_DIR, "device_" + short_ip + "_" + ts + ".csv")
        device_files[device_ip] = filename

    output_file = device_files[device_ip]
    try:
        file_exists = os.path.isfile(output_file)
        with open(output_file, 'a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=features.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)
    except Exception as e:
        print("  Warning UDP flush error: " + str(e))


def parse_tcpdump_line(line):
    global total_packets, total_flows, dropped_flows

    try:
        parts = line.strip().split()
        if len(parts) < 4:
            return

        timestamp = float(parts[0])

        is_arp = (parts[1] == 'ARP,' or parts[1] == 'ARP')
        is_icmp = False
        is_udp = False

        if not is_arp:
            for p in parts:
                if p == 'ICMP':
                    is_icmp = True
                    break
                if p == 'UDP,' or p == 'UDP':
                    is_udp = True
                    break

        # ── Parse ARP ──
        if is_arp:
            src_ip = "0.0.0.0"
            dst_ip = "0.0.0.0"
            for i, p in enumerate(parts):
                if p == 'who-has' and i + 1 < len(parts):
                    dst_ip = parts[i + 1].rstrip(',')
                if p == 'tell' and i + 1 < len(parts):
                    src_ip = parts[i + 1].rstrip(',')
                if p == 'Reply' and i + 1 < len(parts):
                    src_ip = parts[i + 1].rstrip(',')
                    if 'is-at' in parts:
                        dst_ip = src_ip

            src_port = 0
            dst_port = 0
            header_size = ARP_HEADER_SIZE
            proto_num = 0  # ARP
            flags_str = ""
            device_ip = src_ip

            pkt_len = 28  # ARP is fixed size
            total_len = pkt_len + header_size

            flow_key = (src_ip, 0, dst_ip, 0)
            direction = 'fwd'
            update_port_stats(0, total_len, 'fwd')

        # ── Parse ICMP ──
        elif is_icmp:
            src_full = parts[2]
            dst_full = parts[4].rstrip(':')

            src_ip = src_full
            dst_ip = dst_full
            src_port = 0
            dst_port = 0
            header_size = ICMP_HEADER_SIZE
            proto_num = 1  # ICMP
            flags_str = ""

            pkt_len = 0
            for i, p in enumerate(parts):
                if p == 'length':
                    if i + 1 < len(parts):
                        try:
                            pkt_len = int(parts[i + 1].rstrip(',:'))
                        except ValueError:
                            pass
                    break

            total_len = pkt_len + header_size

            # Skip ICMP responses from RPi (port unreachable etc)
            if src_ip == PI_IP:
                return

            # ICMP: use packet window buffer
            if dst_ip == PI_IP:
                device_ip = dst_ip if DDOS_MODE else src_ip
                if RECON_MODE:
                    buf_key = ('recon', src_ip, dst_ip)
                elif DDOS_MODE:
                    buf_key = ('ddos', dst_ip, 0)
                else:
                    buf_key = (src_ip, dst_ip, 0)
            else:
                device_ip = dst_ip
                buf_key = (dst_ip, src_ip, 0)

            with lock:
                if buf_key not in udp_buffers:
                    udp_buffers[buf_key] = []
                udp_buffers[buf_key].append({
                    'timestamp': timestamp, 'pkt_len': total_len,
                    'src_ip': src_ip, 'dst_ip': dst_ip,
                    'dst_port': 0, 'header_size': header_size,
                    'is_icmp': True
                })
                if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                    pkts = udp_buffers.pop(buf_key, [])
                    fi = FlowInfo(pkts[0]['src_ip'], 0, pkts[0]['dst_ip'], 0)
                    fi.protocol = 1  # ICMP
                    fi.start_time = pkts[0]['timestamp']
                    fi.end_time = pkts[-1]['timestamp']
                    for pkt in pkts:
                        fi.fwd_packets += 1
                        fi.fwd_bytes += pkt['pkt_len']
                        fi.fwd_header_bytes += pkt['header_size']
                        if len(fi.fwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                            fi.fwd_pkt_sizes.append(pkt['pkt_len'])
                            fi.fwd_times.append(pkt['timestamp'])
                    features = extract_features(fi)
                    if features:
                        features['service'] = 'ICMP'
                        if device_ip not in device_files:
                            ts_str = datetime.now().strftime("%d%m")
                            short_ip = device_ip.split('.')[-1]
                            fname = os.path.join(FEATURES_DIR, "device_" + short_ip + "_" + ts_str + ".csv")
                            device_files[device_ip] = fname
                        output_file = device_files[device_ip]
                        try:
                            file_exists = os.path.isfile(output_file)
                            with open(output_file, 'a', newline='') as f_out:
                                writer = csv.DictWriter(f_out, fieldnames=features.keys())
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(features)
                        except Exception:
                            pass

            update_port_stats(0, total_len, 'fwd')
            return

        # ── Parse TCP / UDP ──
        else:
            header_size = UDP_HEADER_SIZE if is_udp else TCP_HEADER_SIZE
            proto_num = 17 if is_udp else 6

            src_full = parts[2]
            dst_full = parts[4].rstrip(':')

            src_parts = src_full.rsplit('.', 1)
            if len(src_parts) != 2:
                return
            src_ip = src_parts[0]
            src_port = int(src_parts[1])

            dst_parts = dst_full.rsplit('.', 1)
            if len(dst_parts) != 2:
                return
            dst_ip = dst_parts[0]
            dst_port = int(dst_parts[1])

            # Extract flags (TCP only)
            flags_str = ""
            if not is_udp:
                for p in parts:
                    if p.startswith('Flags'):
                        idx = parts.index(p)
                        if idx + 1 < len(parts):
                            flags_str = parts[idx + 1].strip('[].,')
                        break

            # Extract length
            pkt_len = 0
            for i, p in enumerate(parts):
                if p == 'length':
                    if i + 1 < len(parts):
                        try:
                            pkt_len = int(parts[i + 1].rstrip(',:'))
                        except ValueError:
                            pass
                    break

            total_len = pkt_len + header_size

            # MITM_MODE: capture all TCP to Pi:5000 from any source.
            # Label is assigned post-capture by src_ip using label_mitm_csv.py:
            #   src in SENSOR_IPS  → MitM_ARP        (VM1 intercepted sensor traffic)
            #   src == 150         → MitM_ARP        (VM1 direct)
            #   src == 156         → Spoofing_Sensor (VM2 impersonating sensor)
            #   src == 170         → Replay_Sensor   (VM3 replaying packets)
            if MITM_MODE and dst_port == 5000 and dst_ip == PI_IP:
                # Use 10-packet window buffer — same as UDP/ICMP/DDoS modes.
                # This works for:
                #   - Normal TCP sessions (FIN/RST may never arrive with spoofed src)
                #   - Scapy raw packets with spoofed src_ip (no TCP handshake)
                #   - Replay packets
                # Groups by src_ip only — one buffer per attacker/sensor IP
                device_ip = src_ip
                buf_key = ('mitm', src_ip, dst_port)

                with lock:
                    if buf_key not in udp_buffers:
                        udp_buffers[buf_key] = []
                        print("\n🕵 MitM flow: " + src_ip +
                              " → " + dst_ip + ":" + str(dst_port))

                    udp_buffers[buf_key].append({
                        'timestamp':   timestamp,
                        'pkt_len':     total_len,
                        'src_ip':      src_ip,
                        'dst_ip':      dst_ip,
                        'dst_port':    dst_port,
                        'header_size': header_size,
                        'flags':       flags_str,
                    })

                    if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                        # Flush buffer → write one CSV row per 10 packets
                        pkts = udp_buffers.pop(buf_key, [])
                        fi = FlowInfo(src_ip, 0, dst_ip, dst_port)
                        fi.protocol = 6  # TCP
                        fi.start_time = pkts[0]['timestamp']
                        fi.end_time   = pkts[-1]['timestamp']

                        for pkt in pkts:
                            fi.fwd_packets += 1
                            fi.fwd_bytes   += pkt['pkt_len']
                            fi.fwd_header_bytes += pkt['header_size']
                            if len(fi.fwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                                fi.fwd_pkt_sizes.append(pkt['pkt_len'])
                                fi.fwd_times.append(pkt['timestamp'])
                            f = pkt.get('flags', '')
                            if 'S' in f: fi.syn += 1
                            if 'F' in f: fi.fin += 1
                            if 'R' in f: fi.rst += 1
                            if 'P' in f: fi.psh += 1
                            if '.' in f: fi.ack += 1
                            if 'E' in f: fi.ece += 1
                            if 'W' in f: fi.cwr += 1
                            if 'U' in f: fi.urg += 1

                        features = extract_features(fi)
                        if features:
                            features['service'] = 'Flask/IoT'
                            # Ensure _mitm suffix on output file
                            if device_ip not in device_files:
                                ts = datetime.now().strftime("%d%m")
                                short_ip = device_ip.split('.')[-1]
                                filename = os.path.join(
                                    FEATURES_DIR,
                                    "device_" + short_ip + "_" + ts + "_mitm.csv"
                                )
                                device_files[device_ip] = filename
                            output_file = device_files[device_ip]
                            try:
                                file_exists = os.path.isfile(output_file)
                                with open(output_file, 'a', newline='') as f_out:
                                    writer = csv.DictWriter(f_out, fieldnames=features.keys())
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerow(features)
                            except Exception as e:
                                print("  ⚠ MitM save error: " + str(e))

                update_port_stats(dst_port, total_len, 'fwd')
                return  # handled — skip normal routing
            # ── END MITM MODE ─────────────────────────────────────────────────

            # ── BACKDOOR MODE ─────────────────────────────────────────────────
            # Must be before the PI_IP src filter — Pi initiates outbound connection.
            # Port-agnostic: any TCP between Pi and a C2 VM is backdoor traffic.
            # Covers reverse shells (Meterpreter, netcat, Python) on any port,
            # and exfiltration flows (Pi → VM2 data receiver) on any port.
            if BACKDOOR_MODE:
                is_backdoor = (
                    (src_ip == PI_IP and dst_ip in BACKDOOR_C2_IPS) or
                    (dst_ip == PI_IP and src_ip in BACKDOOR_C2_IPS)
                )
                if is_backdoor:
                    # Always normalize to Pi→C2 direction for flow key
                    # so FIN/ACK from C2→Pi goes into the SAME flow
                    if src_ip == PI_IP:
                        # Pi → C2 (outbound: beacon, exfil)
                        c2_ip   = dst_ip
                        c2_port = dst_port
                    else:
                        # C2 → Pi (inbound: commands, ACK, FIN)
                        c2_ip   = src_ip
                        c2_port = src_port

                    flow_key  = (PI_IP, 0, c2_ip, c2_port)
                    direction = 'fwd' if src_ip == PI_IP else 'bwd'
                    device_ip = PI_IP

                    with lock:
                        if flow_key not in flow_data:
                            # Don't create new flow for FIN/RST — these are
                            # closing packets for already-saved flows
                            if 'F' in flags_str or 'R' in flags_str:
                                return
                            fi = FlowInfo(PI_IP, 0, c2_ip, c2_port)
                            fi.protocol = 6  # TCP
                            flow_data[flow_key] = fi
                            total_flows += 1

                            if device_ip not in device_files:
                                ts = datetime.now().strftime("%d%m")
                                short_ip = device_ip.split('.')[-1]
                                filename = os.path.join(
                                    FEATURES_DIR,
                                    "device_" + short_ip + "_" + ts + "_backdoor.csv"
                                )
                                device_files[device_ip] = filename
                                print("\n🎯 Backdoor session: " + src_ip +
                                      " → " + dst_ip + ":" + str(dst_port))

                        flow = flow_data[flow_key]
                        if flow.start_time is None:
                            flow.start_time = timestamp
                        flow.end_time = timestamp

                        if 'S' in flags_str: flow.syn += 1
                        if 'F' in flags_str: flow.fin += 1
                        if 'R' in flags_str: flow.rst += 1
                        if 'P' in flags_str: flow.psh += 1
                        if '.' in flags_str: flow.ack += 1
                        if 'E' in flags_str: flow.ece += 1
                        if 'W' in flags_str: flow.cwr += 1
                        if 'U' in flags_str: flow.urg += 1

                        if direction == 'fwd':
                            flow.fwd_packets += 1
                            flow.fwd_bytes += total_len
                            flow.fwd_header_bytes += header_size
                            if len(flow.fwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                                flow.fwd_pkt_sizes.append(total_len)
                                flow.fwd_times.append(timestamp)
                        else:
                            flow.bwd_packets += 1
                            flow.bwd_bytes += total_len
                            flow.bwd_header_bytes += header_size
                            if len(flow.bwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                                flow.bwd_pkt_sizes.append(total_len)
                                flow.bwd_times.append(timestamp)

                        total_packets += 1

                        # Save on FIN/RST (session closed)
                        total_fwd = flow.fwd_packets
                        total_bwd = flow.bwd_packets
                        should_save = (
                            ('F' in flags_str or 'R' in flags_str) or  # session end
                            (total_fwd + total_bwd) % 50 == 0           # periodic every 50 pkts
                        )
                        if should_save and (total_fwd + total_bwd) > 0:
                            features = extract_features(flow)
                            if features:
                                features['service'] = 'Backdoor-C2'
                                output_file = device_files[device_ip]
                                try:
                                    file_exists = os.path.isfile(output_file)
                                    with open(output_file, 'a', newline='') as f_out:
                                        writer = csv.DictWriter(f_out, fieldnames=features.keys())
                                        if not file_exists:
                                            writer.writeheader()
                                        writer.writerow(features)
                                except Exception as e:
                                    print("  ⚠ Backdoor save error: " + str(e))
                        if 'F' in flags_str or 'R' in flags_str:
                            del flow_data[flow_key]
                    return  # Handled — skip normal routing
                # Not a backdoor flow — still return to prevent
                # fallthrough to normal routing in BACKDOOR_MODE
                return
            # ── END BACKDOOR MODE ─────────────────────────────────────────────

            # ── MITM MODE ─────────────────────────────────────────────────────
            # Captures three attack types all targeting Pi:5000 sensor endpoint:
            #
            # 1. MitM_ARP:        src=SENSOR_IP via VM3 (ARP spoofed)
            #                     VM3 intercepts and modifies before forwarding
            # 2. Spoofing_Sensor: src=VM3_IP, impersonating a sensor
            #                     dst=Pi:5000, device field contains sensor IP
            # 3. Replay_Sensor:   src=VM3_IP, replaying captured sensor packets
            #                     high rate, identical payloads
            #
            # All three arrive at Pi:5000 — distinguished by src_ip:
            #   src in SENSOR_IPS → MitM_ARP (intercepted sensor traffic)
            #   src == 156         → Spoofing_Sensor (VM2)
            #   src == 170         → Replay_Sensor   (VM3)


            # Skip outgoing traffic from RPi (normal mode)
            if src_ip == PI_IP:
                return

            # RECON_MODE: ALL TCP/UDP goes through 10-packet buffer
            if RECON_MODE:
                buf_key = ('recon', src_ip, dst_ip)
                with lock:
                    if buf_key not in udp_buffers:
                        udp_buffers[buf_key] = []
                    udp_buffers[buf_key].append({
                        'timestamp': timestamp, 'pkt_len': total_len,
                        'src_ip': src_ip, 'dst_ip': dst_ip,
                        'dst_port': dst_port, 'header_size': header_size,
                        'flags': flags_str
                    })
                    if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                        flush_udp_buffer(buf_key, src_ip)
                update_port_stats(dst_port, total_len, 'fwd')
                return

            # GRE_MODE: handle port 0 packets through packet buffer
            if GRE_MODE and dst_port == 0 and dst_ip == PI_IP:
                buf_key = ('gre', dst_ip, 0)
                with lock:
                    if buf_key not in udp_buffers:
                        udp_buffers[buf_key] = []
                    udp_buffers[buf_key].append({
                        'timestamp': timestamp, 'pkt_len': total_len,
                        'src_ip': src_ip, 'dst_ip': dst_ip,
                        'dst_port': 0, 'header_size': header_size,
                        'flags': flags_str
                    })
                    if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                        flush_udp_buffer(buf_key, dst_ip)
                update_port_stats(0, total_len, 'fwd')
                return

            # Skip port 0 unless GRE_MODE or DDOS_MODE (fragmented UDP has port 0)
            if dst_port == 0 and not GRE_MODE and not DDOS_MODE:
                return

            # Determine flow direction
            # In DDOS_MODE, port 0 fragmented UDP goes directly to buffer
            if DDOS_MODE and dst_port == 0 and dst_ip == PI_IP:
                buf_key = ('ddos', dst_ip, 0)
                with lock:
                    if buf_key not in udp_buffers:
                        udp_buffers[buf_key] = []
                    udp_buffers[buf_key].append({
                        'timestamp': timestamp, 'pkt_len': total_len,
                        'src_ip': src_ip, 'dst_ip': dst_ip,
                        'dst_port': 0, 'header_size': header_size,
                        'flags': flags_str
                    })
                    if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                        flush_udp_buffer(buf_key, dst_ip)
                update_port_stats(0, total_len, 'fwd')
                return

            if dst_port in MONITORED_PORTS:
                device_ip = src_ip
                if DDOS_MODE and dst_port not in {2244, 22, 3000, 8086}:
                    buf_key = ('ddos', dst_ip, dst_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': src_ip, 'dst_ip': dst_ip,
                            'dst_port': dst_port, 'header_size': header_size,
                            'flags': flags_str
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, dst_ip)
                    update_port_stats(dst_port, total_len, 'fwd')
                    return
                elif is_udp:
                    buf_key = (src_ip, dst_ip, dst_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': src_ip, 'dst_ip': dst_ip,
                            'dst_port': dst_port, 'header_size': header_size
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, device_ip)
                    update_port_stats(dst_port, total_len, 'fwd')
                    return
                elif DOS_MODE and dst_port not in {2244, 22, 3000, 8086}:
                    buf_key = (src_ip, dst_ip, dst_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': src_ip, 'dst_ip': dst_ip,
                            'dst_port': dst_port, 'header_size': header_size,
                            'flags': flags_str
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, src_ip)
                    update_port_stats(dst_port, total_len, 'fwd')
                    return
                elif src_ip in SPOOF_IPS:
                    buf_key = (src_ip, dst_ip, dst_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': src_ip, 'dst_ip': dst_ip,
                            'dst_port': dst_port, 'header_size': header_size,
                            'flags': flags_str
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, src_ip)
                    update_port_stats(dst_port, total_len, 'fwd')
                    return
                else:
                    flow_key = (src_ip, src_port, dst_ip, dst_port)
                direction = 'fwd'
                update_port_stats(dst_port, total_len, 'fwd')
            elif src_port in MONITORED_PORTS:
                # Skip responses back to RPi (RPi initiated these connections)
                if dst_ip == PI_IP:
                    return
                device_ip = dst_ip
                if DDOS_MODE and src_port not in {2244, 22, 3000, 8086}:
                    buf_key = ('ddos', src_ip, src_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': dst_ip, 'dst_ip': src_ip,
                            'dst_port': src_port, 'header_size': header_size,
                            'flags': flags_str
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, dst_ip)
                    update_port_stats(src_port, total_len, 'bwd')
                    return
                elif is_udp:
                    buf_key = (dst_ip, src_ip, src_port)
                    with lock:
                        if buf_key not in udp_buffers:
                            udp_buffers[buf_key] = []
                        udp_buffers[buf_key].append({
                            'timestamp': timestamp, 'pkt_len': total_len,
                            'src_ip': dst_ip, 'dst_ip': src_ip,
                            'dst_port': src_port, 'header_size': header_size
                        })
                        if len(udp_buffers[buf_key]) >= UDP_PACKET_WINDOW:
                            flush_udp_buffer(buf_key, device_ip)
                    update_port_stats(src_port, total_len, 'bwd')
                    return
                else:
                    flow_key = (dst_ip, dst_port, src_ip, src_port)
                direction = 'bwd'
                update_port_stats(src_port, total_len, 'bwd')

            else:
                return

        with lock:
            if flow_key not in flow_data and len(flow_data) >= EMERGENCY_FLOW_LIMIT:
                dropped_flows += 1
                if dropped_flows % 1000 == 0:
                    print("  ⚠ Dropped " + str(dropped_flows) + " flows (memory limit)", flush=True)
                return

            if flow_key not in flow_data:
                if is_icmp:
                    if dst_ip == PI_IP:
                        fi = FlowInfo(src_ip, 0, dst_ip, 0)
                    else:
                        fi = FlowInfo(dst_ip, 0, src_ip, 0)
                elif is_udp and dst_port in MONITORED_PORTS:
                    fi = FlowInfo(src_ip, 0, dst_ip, dst_port)
                elif is_udp and src_port in MONITORED_PORTS:
                    fi = FlowInfo(dst_ip, 0, src_ip, src_port)
                else:
                    fi = FlowInfo(
                        flow_key[0], flow_key[1],
                        flow_key[2], flow_key[3]
                    )
                flow_data[flow_key] = fi
                total_flows += 1

                if device_ip not in device_files:
                    ts = datetime.now().strftime("%d%m")
                    short_ip = device_ip.split('.')[-1]
                    filename = os.path.join(FEATURES_DIR, "device_" + short_ip + "_" + ts + ".csv")
                    device_files[device_ip] = filename
                    print("\n📱 New device: " + device_ip + " → " + filename)

            flow = flow_data[flow_key]
            flow.protocol = proto_num

            if flow.start_time is None:
                flow.start_time = timestamp
            flow.end_time = timestamp

            # Count all flags (TCP only — UDP/ICMP/ARP have no TCP flags)
            if proto_num == 6:  # TCP
                if 'S' in flags_str:
                    flow.syn += 1
                if 'F' in flags_str:
                    flow.fin += 1
                if 'R' in flags_str:
                    flow.rst += 1
                if 'P' in flags_str:
                    flow.psh += 1
                if '.' in flags_str:
                    flow.ack += 1
                if 'E' in flags_str:
                    flow.ece += 1
                if 'W' in flags_str:
                    flow.cwr += 1
                if 'U' in flags_str:
                    flow.urg += 1

            if direction == 'fwd':
                flow.fwd_packets += 1
                flow.fwd_bytes += total_len
                flow.fwd_header_bytes += header_size
                if len(flow.fwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                    flow.fwd_pkt_sizes.append(total_len)
                    flow.fwd_times.append(timestamp)
            else:
                flow.bwd_packets += 1
                flow.bwd_bytes += total_len
                flow.bwd_header_bytes += header_size
                if len(flow.bwd_pkt_sizes) < MAX_PKT_SIZES_PER_FLOW:
                    flow.bwd_pkt_sizes.append(total_len)
                    flow.bwd_times.append(timestamp)

            total_packets += 1

            # For web ports: save and remove flow on FIN or RST
            # This gives per-request granularity instead of per-flow
            service_port = flow.service_port
            if proto_num == 6 and service_port in WEB_PORTS:
                if ('F' in flags_str or 'R' in flags_str) and flow.fwd_packets > 0:
                    features = extract_features(flow)
                    if features:
                        device_ip_key = flow_key[0] if direction == 'fwd' else flow_key[2]
                        if device_ip_key not in device_files:
                            ts = datetime.now().strftime("%d%m")
                            short_ip = device_ip_key.split('.')[-1]
                            filename = os.path.join(FEATURES_DIR, "device_" + short_ip + "_" + ts + ".csv")
                            device_files[device_ip_key] = filename
                        output_file = device_files[device_ip_key]
                        try:
                            file_exists = os.path.isfile(output_file)
                            with open(output_file, 'a', newline='') as f_out:
                                writer = csv.DictWriter(f_out, fieldnames=features.keys())
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(features)
                        except Exception:
                            pass
                    del flow_data[flow_key]

    except Exception:
        pass


def update_port_stats(port, pkt_len, direction):
    if port not in port_stats:
        port_stats[port] = {
            'packets_in': 0, 'packets_out': 0,
            'bytes_in': 0, 'bytes_out': 0
        }
    if direction == 'fwd':
        port_stats[port]['packets_in'] += 1
        port_stats[port]['bytes_in'] += pkt_len
    else:
        port_stats[port]['packets_out'] += 1
        port_stats[port]['bytes_out'] += pkt_len


# ─────────────────────────────────────────────
# Feature extraction (CIC-IoT-2023 superset)
# ─────────────────────────────────────────────
def extract_features(flow):
    duration_us = (flow.end_time - flow.start_time) * 1_000_000 if flow.end_time and flow.start_time else 0
    duration_s = duration_us / 1_000_000 if duration_us > 0 else 0

    # Skip zero-packet flows only
    total_pkts_early = flow.fwd_packets + flow.bwd_packets
    if total_pkts_early == 0:
        return None

    fwd_lens = flow.fwd_pkt_sizes
    bwd_lens = flow.bwd_pkt_sizes
    all_lens = fwd_lens + bwd_lens

    total_pkts = flow.fwd_packets + flow.bwd_packets
    total_bytes = flow.fwd_bytes + flow.bwd_bytes
    header_total = flow.fwd_header_bytes + flow.bwd_header_bytes

    # IATs
    all_times = sorted(flow.fwd_times + flow.bwd_times)
    all_iats = []
    for i in range(len(all_times) - 1):
        all_iats.append((all_times[i+1] - all_times[i]) * 1_000_000)

    fwd_iats = []
    for i in range(len(flow.fwd_times) - 1):
        fwd_iats.append((flow.fwd_times[i+1] - flow.fwd_times[i]) * 1_000_000)

    bwd_iats = []
    for i in range(len(flow.bwd_times) - 1):
        bwd_iats.append((flow.bwd_times[i+1] - flow.bwd_times[i]) * 1_000_000)

    # Rates
    rate = total_pkts / duration_s if duration_s > 0 else 0
    srate = flow.fwd_packets / duration_s if duration_s > 0 else 0
    drate = flow.bwd_packets / duration_s if duration_s > 0 else 0

    # Protocol booleans (CIC-IoT-2023 style)
    service_port = flow.service_port
    proto = PORT_PROTOCOL.get(service_port, 'TCP')

    features = {
        # ── Metadata (our extras) ──
        'timestamp': datetime.fromtimestamp(flow.start_time).isoformat() if flow.start_time else '',
        'src_ip': flow.src_ip,
        'src_port': flow.src_port,
        'dst_ip': flow.dst_ip,
        'dst_port': flow.dst_port,
        'service': 'ICMP' if flow.protocol == 1 else ('ARP' if flow.protocol == 0 else PORT_LABELS.get(service_port, "port_" + str(service_port))),

        # ── CIC-IoT-2023 features ──
        'flow_duration': duration_us,
        'Header_Length': header_total,
        'Protocol_Type': flow.protocol,  # 6=TCP, 17=UDP, 1=ICMP, 0=ARP
        'Duration': duration_s,
        'Rate': rate,
        'Srate': srate,
        'Drate': drate,
        'fin_flag_number': flow.fin,
        'syn_flag_number': flow.syn,
        'rst_flag_number': flow.rst,
        'psh_flag_number': flow.psh,
        'ack_flag_number': flow.ack,
        'ece_flag_number': flow.ece,
        'cwr_flag_number': flow.cwr,
        'ack_count': flow.ack,
        'syn_count': flow.syn,
        'fin_count': flow.fin,
        'urg_count': flow.urg,
        'rst_count': flow.rst,
        'HTTP': 1 if proto == 'HTTP' else 0,
        'HTTPS': 1 if proto == 'HTTPS' else 0,
        'DNS': 1 if service_port == 53 else 0,
        'Telnet': 1 if proto == 'Telnet' else 0,
        'SMTP': 1 if proto == 'SMTP' else 0,
        'SSH': 1 if proto == 'SSH' else 0,
        'IRC': 1 if service_port == 6667 else 0,
        'TCP': 1 if flow.protocol == 6 else 0,
        'UDP': 1 if flow.protocol == 17 else 0,
        'DHCP': 1 if service_port in (67, 68) else 0,
        'ARP': 1 if flow.protocol == 0 else 0,
        'ICMP': 1 if flow.protocol == 1 else 0,
        'IPv': 1 if flow.protocol in (6, 17, 1) else 0,
        'LLC': 0,  # LLC is layer 2 framing — not detectable from tcpdump IP layer
        'Tot_sum': total_bytes,
        'Min': min(all_lens) if all_lens else 0,
        'Max': max(all_lens) if all_lens else 0,
        'AVG': mean(all_lens),
        'Std': std(all_lens),
        'Tot_size': total_bytes,
        'IAT': mean(all_iats),
        'Number': total_pkts,
        'Magnitude': magnitude(fwd_lens, bwd_lens),
        'Radius': radius(fwd_lens, bwd_lens),
        'Covariance': covariance(fwd_lens, bwd_lens),
        'Variance': variance(all_lens),
        'Weight': total_pkts * duration_s if duration_s > 0 else 0,

        # ── Our extra features (not in CIC) ──
        'total_fwd_packets': flow.fwd_packets,
        'total_bwd_packets': flow.bwd_packets,
        'total_length_fwd_packets': flow.fwd_bytes,
        'total_length_bwd_packets': flow.bwd_bytes,
        'fwd_packet_length_mean': mean(fwd_lens),
        'fwd_packet_length_max': max(fwd_lens) if fwd_lens else 0,
        'fwd_packet_length_min': min(fwd_lens) if fwd_lens else 0,
        'fwd_packet_length_std': std(fwd_lens),
        'bwd_packet_length_mean': mean(bwd_lens),
        'bwd_packet_length_max': max(bwd_lens) if bwd_lens else 0,
        'bwd_packet_length_min': min(bwd_lens) if bwd_lens else 0,
        'bwd_packet_length_std': std(bwd_lens),
        'flow_bytes_s': total_bytes / duration_s if duration_s > 0 else 0,
        'flow_packets_s': rate,
        'fwd_iat_mean': mean(fwd_iats),
        'fwd_iat_max': max(fwd_iats) if fwd_iats else 0,
        'fwd_iat_min': min(fwd_iats) if fwd_iats else 0,
        'fwd_iat_std': std(fwd_iats),
        'bwd_iat_mean': mean(bwd_iats),
        'bwd_iat_max': max(bwd_iats) if bwd_iats else 0,
        'bwd_iat_min': min(bwd_iats) if bwd_iats else 0,
        'bwd_iat_std': std(bwd_iats),
        'down_up_ratio': flow.bwd_packets / flow.fwd_packets if flow.fwd_packets > 0 else 0,
    }

    return features


# ─────────────────────────────────────────────
# InfluxDB writers
# ─────────────────────────────────────────────
def write_flows_to_influx(features_list):
    if not influx_client or not features_list:
        return

    json_body = []
    for f in features_list:
        port_label = PORT_LABELS.get(f['dst_port'], "port_" + str(f['dst_port']))
        entry = {
            "measurement": "network_flows",
            "tags": {
                "src_ip": f['src_ip'],
                "dst_ip": f['dst_ip'],
                "dst_port": str(f['dst_port']),
                "service": port_label,
            },
            "fields": {
                "flow_duration": float(f['flow_duration']),
                "total_fwd_packets": int(f['total_fwd_packets']),
                "total_bwd_packets": int(f['total_bwd_packets']),
                "total_packets": int(f['Number']),
                "total_length_fwd": int(f['total_length_fwd_packets']),
                "total_length_bwd": int(f['total_length_bwd_packets']),
                "fwd_pkt_len_mean": float(f['fwd_packet_length_mean']),
                "bwd_pkt_len_mean": float(f['bwd_packet_length_mean']),
                "flow_bytes_s": float(f['flow_bytes_s']),
                "flow_packets_s": float(f['flow_packets_s']),
                "syn_count": int(f['syn_flag_number']),
                "fin_count": int(f['fin_flag_number']),
                "rst_count": int(f['rst_flag_number']),
                "psh_count": int(f['psh_flag_number']),
                "ack_count": int(f['ack_flag_number']),
                "down_up_ratio": float(f['down_up_ratio']),
                "packet_length_mean": float(f['AVG']),
                "packet_length_std": float(f['Std']),
                "Rate": float(f['Rate']),
                "Srate": float(f['Srate']),
                "Drate": float(f['Drate']),
                "Magnitude": float(f['Magnitude']),
                "Radius": float(f['Radius']),
                "Variance": float(f['Variance']),
            }
        }
        json_body.append(entry)

    try:
        influx_client.write_points(json_body)
    except Exception as e:
        print("  ⚠ InfluxDB write error: " + str(e))


def write_port_stats_to_influx():
    if not influx_client:
        return

    json_body = []
    for port in port_stats:
        port_label = PORT_LABELS.get(port, "port_" + str(port))
        json_body.append({
            "measurement": "port_stats",
            "tags": {"port": str(port), "service": port_label},
            "fields": {
                "packets_in": port_stats[port]['packets_in'],
                "packets_out": port_stats[port]['packets_out'],
                "bytes_in": port_stats[port]['bytes_in'],
                "bytes_out": port_stats[port]['bytes_out'],
            }
        })

    if json_body:
        try:
            influx_client.write_points(json_body)
        except Exception:
            pass


# ─────────────────────────────────────────────
# Save & purge
# ─────────────────────────────────────────────
def save_all_flows():
    global forced_saves

    with lock:
        if not flow_data:
            return

        now = time.time()
        flow_count = len(flow_data)

        if flow_count >= MAX_FLOWS:
            forced_saves += 1
            print("  ⚠ FORCED SAVE — " + str(flow_count) + " flows in memory (limit: " + str(MAX_FLOWS) + ")", flush=True)

        device_features = defaultdict(list)
        for flow_key, flow in flow_data.items():
            is_expired = flow.end_time and (now - flow.end_time) > FLOW_EXPIRE_SECONDS
            if not is_expired and flow_count < MAX_FLOWS:
                continue
            features = extract_features(flow)
            if features:
                device_ip = flow_key[0]
                device_features[device_ip].append(features)

        for device_ip, features_list in device_features.items():
            if device_ip not in device_files:
                ts = datetime.now().strftime("%d%m")
                short_ip = device_ip.split('.')[-1]
                filename = os.path.join(FEATURES_DIR, "device_" + short_ip + "_" + ts + ".csv")
                device_files[device_ip] = filename

            output_file = device_files[device_ip]
            try:
                file_exists = os.path.isfile(output_file)
                with open(output_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=features_list[0].keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(features_list)
            except Exception as e:
                print("  ⚠ CSV save error [" + device_ip + "]: " + str(e))

            write_flows_to_influx(features_list)

        write_port_stats_to_influx()

        total_features = sum(len(fl) for fl in device_features.values())

        expired = [k for k, v in flow_data.items()
                   if v.end_time and (now - v.end_time) > FLOW_EXPIRE_SECONDS]
        for k in expired:
            del flow_data[k]

        if len(flow_data) >= MAX_FLOWS:
            sorted_flows = sorted(flow_data.items(), key=lambda x: x[1].end_time or 0)
            purge_count = len(flow_data) - (MAX_FLOWS // 2)
            for i in range(min(purge_count, len(sorted_flows))):
                del flow_data[sorted_flows[i][0]]
            print("  🗑 Purged " + str(purge_count) + " oldest flows to free memory", flush=True)

        remaining = len(flow_data)
        print("  💾 Saved " + str(total_features) + " flows | Active: " + str(remaining) + " | Expired: " + str(len(expired)) + " | Dropped: " + str(dropped_flows), flush=True)


def auto_save_loop():
    while not stop_capture:
        time.sleep(AUTO_SAVE_INTERVAL)
        if not stop_capture:
            save_all_flows()
        with lock:
            if len(flow_data) >= MAX_FLOWS:
                save_all_flows()


# ─────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────
def print_port_stats():
    if not port_stats:
        return
    print("\n  Port Traffic Summary:")
    header = "  " + "Port".ljust(8) + "Service".ljust(14) + "Pkts In".rjust(10) + "Pkts Out".rjust(10) + "Bytes In".rjust(12) + "Bytes Out".rjust(12)
    print(header)
    print("  " + "-" * 66)
    for port in sorted(port_stats.keys()):
        s = port_stats[port]
        label = PORT_LABELS.get(port, "port_" + str(port))
        row = "  " + str(port).ljust(8) + label.ljust(14)
        row += str(s['packets_in']).rjust(10) + str(s['packets_out']).rjust(10)
        row += str(s['bytes_in']).rjust(12) + str(s['bytes_out']).rjust(12)
        print(row)


def print_memory_status():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            info[parts[0].rstrip(':')] = int(parts[1])
        avail = info.get('MemAvailable', 0) // 1024
        total = info.get('MemTotal', 0) // 1024
        used = total - avail
        print("  🧠 RAM: " + str(used) + "MB / " + str(total) + "MB (available: " + str(avail) + "MB)", flush=True)
    except:
        pass


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    global stop_capture

    port_info = ', '.join(str(p) + ' (' + PORT_LABELS.get(p, '?') + ')' for p in MONITORED_PORTS)

    print("=" * 70)
    print("IoT Network Monitor - CIC-IoT-2023 Compatible")
    print("=" * 70)
    print("Interface:       " + INTERFACE)
    print("Filter:          " + CAPTURE_FILTER)
    if BACKDOOR_MODE:
        print("Mode:            BACKDOOR (all TCP ↔ C2 IPs: " + ", ".join(sorted(BACKDOOR_C2_IPS)) + ")")
    elif MITM_MODE:
        print("Mode:            MITM (all TCP → Pi:5000)")
        print("  VM1 (150) → MitM_ARP | VM2 (156) → Spoofing_Sensor | VM3 (170) → Replay_Sensor")
        print("  Sensor IPs:    " + ", ".join(sorted(SENSOR_IPS)))
    elif RECON_MODE:
        print("Mode:            RECON (all ports)")
    elif GRE_MODE:
        print("Mode:            GRE (all dst traffic)")
    else:
        print("Monitored ports: " + port_info)
    print("Auto-save every: " + str(AUTO_SAVE_INTERVAL) + "s")
    print("InfluxDB:        " + ("Connected" if influx_client else "Disabled (CSV only)"))
    print("Output:          " + FEATURES_DIR)
    print("-" * 70)
    print("Features:        CIC-IoT-2023 (46) + extras (20) = 66 columns")
    print("Protocols:       TCP + UDP")
    print("MEMORY LIMITS:")
    print("  Max flows:     " + str(MAX_FLOWS))
    print("  Flow expire:   " + str(FLOW_EXPIRE_SECONDS) + "s")
    print("  Emergency cap: " + str(EMERGENCY_FLOW_LIMIT))
    print("-" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    signal.signal(signal.SIGINT, signal_handler)

    saver = threading.Thread(target=auto_save_loop, daemon=True)
    saver.start()

    print_memory_status()
    print("\n🎯 Starting tcpdump capture...")
    print("📡 Waiting for traffic... (Press Ctrl+C to stop)\n")

    cmd = [
        'tcpdump',
        '-i', INTERFACE,
        '-nn',
        '-tt',
        '-l',
        '--immediate-mode',
        CAPTURE_FILTER
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        status_counter = 0
        while not stop_capture:
            line = proc.stdout.readline()
            if not line:
                break
            parse_tcpdump_line(line)

            if total_packets > 0 and total_packets % 100 == 0:
                status_counter += 1
                status = "  📊 " + str(total_packets) + " packets, " + str(total_flows) + " flows"
                if dropped_flows > 0:
                    status += " (dropped: " + str(dropped_flows) + ")"
                print(status, flush=True)

                if status_counter % 5 == 0:
                    print_memory_status()

    except FileNotFoundError:
        print("❌ tcpdump not found. Install with: sudo apt install tcpdump")
    except KeyboardInterrupt:
        pass
    finally:
        stop_capture = True
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except:
            proc.kill()

    print("\n📦 Final save...")
    save_all_flows()

    print("\n" + "=" * 70)
    print("✓ Capture complete!")
    print("  Total packets:   " + str(total_packets))
    print("  Total flows:     " + str(total_flows))
    print("  Dropped flows:   " + str(dropped_flows))
    print("  Forced saves:    " + str(forced_saves))
    print_port_stats()
    print_memory_status()
    print("\n  Device files:")
    for ip, path in sorted(device_files.items()):
        print("    " + ip + " → " + path)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()