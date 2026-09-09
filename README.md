# Data-Driven Localization of Magneto-Mechanical Resonators from Time Signals

This repository contains the implementation of a deep learning approach for rapidly estimating the position of magneto-mechanical resonators (MMRs) from time-domain signals. MMRs are miniaturized sensor platforms enabling effective tracking and sensing for biomedical applications, but extracting sensor parameters from raw signals is computationally challenging. We demonstrate that a convolutional-transformer architecture trained on simulated data with recorded noise can generalize to experimental measurements, achieving a median localization error of 7 mm with inference in 13 ms per batch, enabling real-time applications.

![Inference results for end-to-end MMR localization](assets/results.png)

## Installation

The code has been developed and tested on AlmaLinux 9.6.

Before proceeding, ensure you have the following software installed:
- [Git](https://git-scm.com/install)
- [Conda](https://github.com/conda-forge/miniforge)

Once the prerequisites are met, you can set up the environment and install the necessary dependencies by running the following commands in your terminal:
```bash
git clone https://github.com/IBIResearch/mmr-learned-localization-from-signals.git
conda env create --file env/python/environment.yml
conda activate mmr-learned-localization-from-signals
pip install -e env/python/
```

Download the data from the [here](https://doi.org/10.15480/882.18147), place it in the `data/raw` directory, and unflatten thefolder structure by running:
```bash
cd data/raw
./flatten.sh decode .
```
You should end up with the following structure inside `data/raw`:
```
.
├── experiment
│   ├── data.npy
│   └── gt.csv
├── flatten.sh
├── model
│   └── checkpoint.pt
├── README.md
└── training
    ├── noise.npy
    ├── test_chunk1.npy
    ├── test_meta.csv
    ├── train_chunk01.npy
    ├── train_chunk02.npy
    ├── train_chunk03.npy
    ├── train_chunk04.npy
    ├── train_chunk05.npy
    ├── train_chunk06.npy
    ├── train_chunk07.npy
    ├── train_chunk08.npy
    ├── train_chunk09.npy
    ├── train_chunk10.npy
    ├── train_meta.csv
    ├── val_chunk1.npy
    └── val_meta.csv
```

## Training

To train a model from scratch, run:
```bash
python src/training/main.py
```
Logs and checkpoints are saved under `data/training/models`.

## Evaluation

To run inference using the provided model, run:
```bash
python src/evaluation/infer.py
python src/evaluation/analysis.py
```
The results can be found in `data/evaluation`.

## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{tsanda_data_driven_mmr_localization_2026,
  author={Tsanda, Artyom and Scheffler, Konrad and Reiss, Sarah and Faltinath, Jonas and Bach, Janik and Thieben, Florian and Stagge, Pascal and Mohn, Fabian and Knopp, Tobias},
  booktitle={2026 IEEE SENSORS},
  title={Data-Driven Localization of Magneto-Mechanical Resonators from Time Signals},
  year={2026},
  volume={},
  number={},
  pages={1-4},
}

```

## License
The code is licensed under the MIT License; see the LICENSE file.
