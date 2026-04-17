import time
import os
import pynvml

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def main(interval_seconds=1.0):
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount() 

        while True:
            clear_console()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i) 
                name = pynvml.nvmlDeviceGetName(handle)  
                if isinstance(name, bytes):  
                    name = name.decode("utf-8", errors="replace") 
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle) 
                print(f"GPU {i} ({name}): {power_mw / 1000:.2f} W")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nSaliendo...")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main(interval_seconds=1.0)
    