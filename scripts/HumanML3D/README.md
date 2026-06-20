# HumanML3D Data Pre-processing

This folder contains an adapter script that converts HumanML3D data into PKL files compatible with LaMoGen utilities.

## Data Structure

| Path                  | Description                                     |
|-----------------------|-------------------------------------------------|
| `workspace/HumanML3D` | Local path to pre-processed HumanML3D data      |
| `assets/HML3D`        | Combined HumanML3D motion data in pickle format |
| `assets/HML3D_lbn`    | Laban-related data derived from HumanML3D       |

## Steps

1. Ensure you are at the **LaMoGen repository root** before running any commands.

2. Clone the official HumanML3D repository into this directory:

   ```bash
   cd scripts/HumanML3D
   git clone https://github.com/EricGuo5513/HumanML3D
   ```

3. Prepare the HumanML3D data (e.g., `new_joint_vecs`, `new_joints`, `texts`, and split files) following the official HumanML3D instructions. For example, place the data under `workspace/HumanML3D`.

4. Run the adapter script:

   ```bash
   python prepare_train_and_val_pkl.py
   ```

5. Run the following script to download the T2M evaluator: 
   
   ```bash
   cd assets
   sh download_glove.sh
   sh download_t2m_evaluators.sh
   ```

Upon completion, the following files will be generated under the LaMoGen repository:

```
assets/HML3D/train.pkl
assets/HML3D/val.pkl
assets/HML3D/test.pkl
```

### Note on Skeleton Offsets

The script requires a single joints `.npy` file to infer target skeleton offsets. By default, it searches under `workspace/HumanML3D/joints` and `workspace/HumanML3D/new_joints`.

### Tips

When running HumanML3D's official MEAN/STD verification scripts (`motion_representation.ipynb`, `cal_mean_variance.ipynb`), the computed MEAN/STD values may differ slightly from the original paper. This is expected and acknowledged by the HumanML3D authors — see [this discussion](https://github.com/EricGuo5513/HumanML3D/issues/3#issuecomment-1173286040) for details.
