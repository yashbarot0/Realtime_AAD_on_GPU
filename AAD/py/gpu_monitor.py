import threading
import time
from pynvml import *
import psutil

class GPUMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.running = False
        self.stats = {
            'gpu_utilization': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 0,
            'memory_utilization': 0,
            'temperature': 0,
            'power_draw': 0
        }
        
        # Initialize NVML
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_id)
            self.available = True
        except:
            print("Warning: NVIDIA GPU monitoring not available")
            self.available = False
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        if not self.available:
            return self.stats
        
        try:
            # GPU utilization
            util = nvmlDeviceGetUtilizationRates(self.handle)
            self.stats['gpu_utilization'] = util.gpu
            
            # Memory usage
            mem_info = nvmlDeviceGetMemoryInfo(self.handle)
            self.stats['memory_used_mb'] = mem_info.used / (1024 * 1024)
            self.stats['memory_total_mb'] = mem_info.total / (1024 * 1024)
            self.stats['memory_utilization'] = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            self.stats['temperature'] = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
            
            # Power draw
            self.stats['power_draw'] = nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
            
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        
        return self.stats.copy()
    
    def start_monitoring(self, interval=1.0):
        """Start background monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval):
        """Background monitoring loop"""
        while self.running:
            self.get_gpu_stats()
            time.sleep(interval)
