"""
iot_device.py
=============
IoT device simulator for REA-HID benchmarking framework.

Attack roles:
  benign   - Normal IoT sensor (drifts after 3600s)
  flood    - Micro-flood burst attack
  slowpub  - Slow telemetry publisher (low-rate pacing attack)
             NOTE: This is a LOW-RATE PACING attack, distinct from the
             original SlowITe (Vaccari et al., Sensors 2020) which exploits
             MQTT KeepAlive broker slot exhaustion. Our attack evaluates
             temporal IAT-based evasion, not connection exhaustion.
  jitter   - Jitter-based evasion (hides in benign IAT shadow)
  sparse   - Sparse publisher (very long IAT, rate-limit evasion)
"""
import asyncio
import random
import json
import time
import os
from paho.mqtt import client as mqtt_client

BROKER    = os.environ.get("BROKER_IP", "127.0.0.1")
PORT      = 1883
DEVICE_ID = os.environ.get("DEVICE_ID", "unknown_device")
ROLE      = os.environ.get("ROLE", "benign")


async def main():
    client = mqtt_client.Client(client_id=DEVICE_ID, protocol=mqtt_client.MQTTv311)

    # FIX 1: 600s KeepAlive prevents background PINGREQ packets from
    # corrupting the true IAT gaps between application-layer messages.
    connected = False
    while not connected:
        try:
            client.connect(BROKER, PORT, keepalive=600)
            connected = True
        except ConnectionRefusedError:
            print(f"[{ROLE.upper()}] {DEVICE_ID}: Broker not ready, retrying in 2s...")
            await asyncio.sleep(2)

    client.loop_start()
    print(f"[{ROLE.upper()}] {DEVICE_ID} connected to {BROKER}")

    # FIX 2: Desynchronized warm-up — random delay up to 180s prevents
    # artificial flow collisions and ensures organic traffic interleaving.
    warmup_delay = random.uniform(0, 180)
    print(f"[{ROLE.upper()}] {DEVICE_ID}: warmup sleep {warmup_delay:.1f}s")
    await asyncio.sleep(warmup_delay)

    start_time = time.time()

    while True:
        try:
            elapsed = time.time() - start_time

            # FIX 3: Temporal Concept Drift Engine
            # After 1 hour, benign devices shift their IAT distribution.
            # This is the exact signal the Page-Hinkley detector must catch.
            if elapsed > 3600 and ROLE == "benign":
                benign_sleep = random.uniform(50.0, 90.0)   # post-drift: slower
            else:
                benign_sleep = random.uniform(30.0, 60.0)   # pre-drift: normal

            # FIX 4: Strict Fixed-Length Payload (Volumetric Camouflage)
            # value_str is always 5 chars; pad is always 32 'X's.
            # Result: all payloads are identical byte length (~95 bytes).
            # This blinds any ML model using packet size as a feature.
            value_str = f"{random.uniform(20.0, 30.0):05.2f}"
            payload   = json.dumps({"v": value_str, "pad": "X" * 32})

            # FIX 5: QoS 0 removes TCP ACK round-trips from IAT calculation.
            if ROLE == "flood":
                # MICRO-FLOOD: burst of 4 packets, then long silence.
                # Total ~18 packets/120s window — volumetrically matches benign.
                # Temporally creates burst density detectable via stddev_piat.
                for _ in range(4):
                    client.publish("sensors/telemetry", payload, qos=0)
                    await asyncio.sleep(0.5)
                sleep_time = random.uniform(150.0, 180.0)

            else:
                client.publish("sensors/telemetry", payload, qos=0)

                if ROLE == "benign":
                    sleep_time = benign_sleep

                elif ROLE == "jitter":
                    # Jitter hides in the original benign IAT shadow (45±15s).
                    # CRITICAL: It does NOT adapt when benign drifts at 3600s.
                    # This creates a temporal divergence the PH detector catches.
                    sleep_time = 45.0 + random.uniform(-15.0, 15.0)

                elif ROLE == "slowpub":
                    # Low-rate pacing attack: 60-120s IAT.
                    # Distinct from Vaccari et al. SlowITe (connection exhaustion).
                    # Our attack targets IAT-based IDS evasion via sparse publishing.
                    sleep_time = random.uniform(60.0, 120.0)

                elif ROLE == "sparse":
                    # Ultra-sparse: 120-300s IAT — extreme rate-limit evasion.
                    sleep_time = random.uniform(120.0, 300.0)

                else:
                    sleep_time = 30.0

            await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] {DEVICE_ID}: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
