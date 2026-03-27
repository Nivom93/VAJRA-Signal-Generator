import numpy as np
from vajra_engine_ultra_v6_final import _kalman_smooth

def test_kalman_dynamic():
    data = np.array([100.0, 105.0, 110.0, 120.0, 200.0, 50.0], dtype=np.float64)
    xhat, vel = _kalman_smooth(data)
    print("xhat:", xhat)
    print("vel:", vel)

if __name__ == '__main__':
    test_kalman_dynamic()
