import ctypes
import numpy as np
from ctypes import Structure, c_double, c_int, c_bool, POINTER

# Load your compiled CUDA library
try:
    # Adjust path based on your library location
    aad_lib = ctypes.CDLL('./libaad_gpu.so')  # Linux
    # aad_lib = ctypes.CDLL('./aad_gpu.dll')  # Windows
except OSError:
    print("Error: Could not load GPU AAD library. Make sure it's compiled and accessible.")
    exit(1)

# C struct definitions matching your AADTypes.h
class BlackScholesParams(Structure):
    _fields_ = [
        ("spot", c_double),
        ("strike", c_double),
        ("time", c_double),
        ("rate", c_double),
        ("volatility", c_double),
        ("is_call", c_bool)
    ]

class OptionResults(Structure):
    _fields_ = [
        ("price", c_double),
        ("delta", c_double),
        ("vega", c_double),
        ("gamma", c_double),
        ("theta", c_double),
        ("rho", c_double)
    ]

class GPUConfig(Structure):
    _fields_ = [
        ("max_tape_size", c_int),
        ("max_scenarios", c_int),
        ("block_size", c_int),
        ("use_fast_math", c_bool)
    ]

# Configure function signatures
aad_lib.launch_blackscholes_kernel.argtypes = [
    POINTER(BlackScholesParams),
    POINTER(OptionResults),
    c_int,
    GPUConfig
]
aad_lib.launch_blackscholes_kernel.restype = None

def compute_option_greeks_gpu(options_data, config=None):
    """
    Compute Greeks for multiple options using GPU AAD
    
    Args:
        options_data: List of dicts with keys: spot, strike, time, rate, volatility, is_call
        config: Optional GPUConfig
    
    Returns:
        tuple: (results_list, num_aad_operations)
    """
    if not options_data:
        return [], 0
    
    num_options = len(options_data)
    
    # Create arrays
    params_array = (BlackScholesParams * num_options)()
    results_array = (OptionResults * num_options)()
    
    # Fill parameters
    for i, opt in enumerate(options_data):
        params_array[i].spot = opt['spot']
        params_array[i].strike = opt['strike']
        params_array[i].time = opt['time']
        params_array[i].rate = opt['rate']
        params_array[i].volatility = opt['volatility']
        params_array[i].is_call = opt.get('is_call', True)
    
    # Configure GPU settings
    if config is None:
        config = GPUConfig()
        config.max_tape_size = 1000
        config.max_scenarios = num_options
        config.block_size = 256
        config.use_fast_math = True
    
    # Call GPU kernel
    aad_lib.launch_blackscholes_kernel(params_array, results_array, num_options, config)
    
    # Convert results
    results = []
    for i in range(num_options):
        results.append({
            'price': results_array[i].price,
            'delta': results_array[i].delta,
            'vega': results_array[i].vega,
            'gamma': results_array[i].gamma,
            'theta': results_array[i].theta,
            'rho': results_array[i].rho
        })
    
    # Estimate AAD operations (each option uses ~100 tape operations)
    aad_operations = num_options * 100
    
    return results, aad_operations
