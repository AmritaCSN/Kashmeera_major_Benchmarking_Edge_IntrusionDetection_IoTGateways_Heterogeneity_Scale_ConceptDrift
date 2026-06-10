import psutil, time, csv

LOG = '/home/dell/rea-hid/rpi_hardware_profile.csv'

def get_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return round(int(f.read().strip()) / 1000, 1)
    except:
        return 0.0

print(f"[PROFILER] Writing to {LOG} — Ctrl+C to stop")
with open(LOG, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t','cpu_pct','ram_used_mb','ram_pct','swap_mb','temp_c'])
    start    = time.time()
    last_msg = -1
    while True:
        cpu  = psutil.cpu_percent(interval=1)
        ram  = psutil.virtual_memory()
        swap = psutil.swap_memory()
        temp = get_temp()
        t    = round(time.time() - start, 1)
        w.writerow([t, cpu,
                    round(ram.used/1e6, 1),
                    round(ram.percent, 1),
                    round(swap.used/1e6, 1),
                    temp])
        f.flush()
        # print every 30s exactly once
        if int(t) % 30 == 0 and int(t) != last_msg and t > 0:
            last_msg = int(t)
            print(f"t={t:.0f}s | CPU={cpu:.1f}% | "
                  f"RAM={ram.used/1e6:.0f}MB ({ram.percent:.1f}%) | "
                  f"Temp={temp}°C")
        time.sleep(1)
