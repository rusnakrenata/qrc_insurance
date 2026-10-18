"""Save and restore the calibrated Aer noise model shared by the three policies of a round."""
import json
from copy import deepcopy

import numpy as np
from qiskit_aer.noise import NoiseModel


def save_noise_model(model, path):
    with open(path, "w") as f:
        json.dump(model.to_dict(serializable=True), f)


def load_noise_model(path):
    """Aer's JSON form stores complex matrices as [re, im] pairs; convert them back."""
    with open(path) as f:
        snapshot = json.load(f)
    decoded = deepcopy(snapshot)
    for error in decoded.get("errors", []):
        for circuit in error.get("instructions", []):
            for inst in circuit:
                if inst["name"] in ("kraus", "unitary"):
                    mats = []
                    for value in inst["params"]:
                        m = np.asarray(value)
                        if m.ndim == 3 and m.shape[-1] == 2:
                            m = m[..., 0] + 1j * m[..., 1]
                        mats.append(m.astype(complex))
                    inst["params"] = mats
    return NoiseModel.from_dict(decoded)
