# BABEL Data Pre-processing

This folder contains an adapter script adapted from [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [BABEL-TEACH](https://github.com/atnikos/teach) to generate PKL files compatible with LaMoGen utilities.

## Background

As described in our main paper, we selected a subset of the HumanML3D dataset that primarily covers locomotion actions such as walking, running, stepping, and jumping. Following the annotation methodology of BABEL-TEACH, these actions were further decomposed into atomic actions and annotated accordingly.

We refer to this subset of HumanML3D with atomic action annotations as **Laban Benchmark (LOCO)**, along with our proposed Laban metrics.

## Data Structure

| Path                    | Description                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------|
| `workspace/babel_teach` | Local path to pre-processed BABEL-TEACH data                                                       |
| `assets/LOCO`           | Filtered locomotion data with flip/normalization augmentations, in HumanML3D motion representation |
| `assets/LOCO_lbn`       | Laban-related data derived from LOCO                                                               |

## Steps

1. Clone the HumanML3D repository first, as the pre-processing code depends on parts of HumanML3D (specifically the forward kinematics module).

2. Download the BABEL-TEACH data from [this link](https://drive.google.com/drive/folders/1gKwLYP8TrbyY1YjsKz1A04s-rCwM9CfX?usp=sharing). This dataset represents the intersection of AMASS and HumanML3D. For example, place the data under `workspace/babel_teach`.

3. Run the adapter script:

   ```bash
   python prepare_train_and_val_pkl.py
   ```

   This converts the original rotation-based motion representation into the HumanML3D motion representation (forward kinematics, canonicalization, velocity computation, etc.) and stores the results in `assets/LOCO`.
