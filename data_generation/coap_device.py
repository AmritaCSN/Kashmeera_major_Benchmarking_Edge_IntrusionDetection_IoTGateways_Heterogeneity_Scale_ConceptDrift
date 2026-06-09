"""
coap_device.py
==============
CoAP client device for the heterogeneity + scale experiment.
Mirrors the behavior of iot_device.py but uses CoAP (UDP port 5683)
instead of MQTT.

Volumetric Camouflage is enforced: payload padded to exactly 95 bytes.
Drift engine mirrors iot_device.py: after 3600s, IAT shifts from
30-60s range to 50-90s range.

Install: pip install aiocoap --break-system-packages
"""

import asyncio
import random
import json
import time
import os

COAP_SERVER = os.environ.get("COAP_SERVER", "192.168.1.100")
DEVICE_ID   = os.environ.get("DEVICE_ID", "coap_unknown")
ROLE        = os.environ.get("ROLE", "coap_client")
PORT        = 5683

async def main():
    try:
        import aiocoap
    except ImportError:
        print(f"[{DEVICE_ID}] aiocoap not installed. Run: pip install aiocoap")
        return

    context = await aiocoap.Context.create_client_context()
    uri = f"coap://{COAP_SERVER}:{PORT}/telemetry"

    # Desynchronized warm-up (mirrors iot_device.py Fix 2)
    warmup = random.uniform(0, 120)
    print(f"[CoAP] {DEVICE_ID}: warming up for {warmup:.1f}s")
    await asyncio.sleep(warmup)

    start_time = time.time()
    print(f"[CoAP] {DEVICE_ID}: connected, sending to {uri}")

    while True:
        elapsed = time.time() - start_time

        # Drift engine (mirrors iot_device.py Fix 3)
        if elapsed > 3600:
            sleep_time = random.uniform(50.0, 90.0)  # post-drift: slower
        else:
            sleep_time = random.uniform(30.0, 60.0)  # pre-drift: normal

        # Fixed payload with volumetric camouflage (mirrors iot_device.py Fix 4)
        value_str  = f"{random.uniform(20.0, 30.0):05.2f}"
        raw_payload = json.dumps({"v": value_str, "pad": "X" * 32}).encode()

        # Pad to exactly 95 bytes
        if len(raw_payload) < 95:
            raw_payload = raw_payload + b"\x00" * (95 - len(raw_payload))
        payload = raw_payload[:95]

        try:
            request = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=payload
            )
            response = await context.request(request).response
        except Exception as e:
            # Don't crash on CoAP errors — just keep sending
            pass

        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
