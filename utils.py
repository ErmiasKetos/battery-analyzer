import numpy as np
from scipy.interpolate import interp1d

def calculate_dqdv_proper(voltage, capacity, num_points=1000):
    """Calculate dQ/dV using proper numerical differentiation"""
    try:
        sort_idx = np.argsort(voltage)
        voltage_sorted = voltage[sort_idx]
        capacity_sorted = capacity[sort_idx]
        
        _, unique_idx = np.unique(voltage_sorted, return_index=True)
        voltage_unique = voltage_sorted[unique_idx]
        capacity_unique = capacity_sorted[unique_idx]
        
        if len(voltage_unique) < 3:
            return None, None
            
        f = interp1d(voltage_unique, capacity_unique, kind='cubic', bounds_error=False)
        v_interp = np.linspace(voltage_unique.min(), voltage_unique.max(), num_points)
        q_interp = f(v_interp)
        dqdv = np.gradient(q_interp, v_interp)
        
        return v_interp, dqdv
        
    except Exception as e:
        print(f"Error in dQ/dV calculation: {str(e)}")
        return None, None
