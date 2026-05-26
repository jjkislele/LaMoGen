# KIT-ML Data Pre-processing

This folder contains an adapter script adapted from HumanML3D to generate PKL files compatible with LaMoGen utilities.

## Data Structure

| Path               | Description                                 |
|--------------------|---------------------------------------------|
| `workspace/KIT-ML` | Local path to pre-processed KIT motion data |
| `assets/KIT`       | Split motion data in pickle format          |
| `assets/KIT_lbn`   | Laban-related data derived from KIT-ML      |

## Steps

1. Download the pre-processed KIT-ML data from [this link](https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK) (as recommended by [HumanML3D](https://github.com/EricGuo5513/HumanML3D)). For example, place the data under `workspace/KIT-ML`.

2. Run the adapter script:

   ```bash
   python prepare_train_and_val_pkl.py
   ```

   This splits and collects the motion data into pickle files stored in `assets/KIT`.
