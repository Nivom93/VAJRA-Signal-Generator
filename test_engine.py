import numpy as np
import vajra_engine_ultra_v6_final as v
import os
import logging
logging.basicConfig(level=logging.INFO)

class DummyConfig:
    pass

class DummyPrecomp:
    def __init__(self):
        self.last_sh = [1.0] * 10
        self.last_sl = [0.9] * 10

cfg = DummyConfig()
brain = v.BrainLearningManager(cfg, long_p='dummy_model', short_p=None)
print("Brains loaded:", brain.brains.keys())

base = {"feature1": [1.0], "feature2": [2.0]}
adv = {}
pre = DummyPrecomp()

try:
    print(f"Features expected: {brain.brains['long']['feature_names']}")
    prob = brain.predict_prob("long", base, adv, 0, 0, 0, 1.0, pre)
    print(f"Predicted Probability for long: {prob}")
    if prob > 0:
        print("Success: Inference using ONNX graph worked properly!")
    else:
        print("Failed: Probability is 0.0")
except Exception as e:
    import traceback
    traceback.print_exc()
