from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ml_mmr_tracking.data import normalization
from ml_mmr_tracking.models import model_factory

DATA_FOLDER = Path("data/raw/experiment")
RESULTS_FOLDER = Path("data/evaluation")
if not RESULTS_FOLDER.exists():
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS = [
    ("deep-conv-transformer", "data/raw/model/checkpoint.pt"),
]

data = np.load(DATA_FOLDER / "data.npy", allow_pickle=True)
results = []
for model_id, checkpoint_path in MODELS:
    model = model_factory.create_model(model_id)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    model.eval()

    for measurement_index in range(data.shape[0]):
        measurement_name = f"M{measurement_index + 1}"
        measurement_data = data[measurement_index]
        measurement_data = np.stack(
            [
                normalization(measurement_data[i].T).T
                for i in range(measurement_data.shape[0])
            ],
            axis=0,
        )

        with torch.no_grad():
            input_tensor = torch.from_numpy(measurement_data).float()
            output = model(input_tensor)
            output *= 0.2  # Scale back to original range

        results.extend(
            [
                (model_id, measurement_name, i + 1, *output[i].numpy().tolist())
                for i in range(output.shape[0])
            ]
        )

df = pd.DataFrame(results, columns=["method", "measurement", "frame", "x", "y", "z"])
df.to_csv(RESULTS_FOLDER / "inference-results.csv", index=False)
