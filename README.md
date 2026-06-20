# LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference

[Junkun Jiang](https://scholar.google.com/citations?user=Oce7VfUAAAAJ&hl=en), [Ho Yin Au](https://github.com/asdryau), [Jingyu Xiang](https://www.linkedin.com/in/jyx-cs/), [Jie Chen](https://scholar.google.com/citations?user=qrWi1RYAAAAJ&hl=en)\*, Hong Kong Baptist University

\* *Corresponding author*

## Abstract

<b>TL;DR</b>
> We introduce LabanLite, a symbolic way to represent human motion that links language and movement. Based on LabanLite, the proposed LaMoGen framework uses LLMs to generate interpretable motions from text. Experiments show it’s more controllable and explainable than previous methods.

<details><summary><b>CLICK for full abstract</b></summary>

> Human motion is highly expressive and naturally aligned with language, yet prevailing methods relying heavily on joint text-motion embeddings struggle to synthesize temporally accurate, detailed motions and often lack explainability. To address these limitations, we introduce LabanLite, a motion representation developed by adapting and extending the Labanotation system. Unlike black-box text–motion embeddings, LabanLite encodes each atomic body-part action (e.g., a single left-foot step) as a discrete Laban symbol paired with a textual template. This abstraction decomposes complex motions into interpretable symbol sequences and body-part instructions, establishing a symbolic link between high-level language and low-level motion trajectories. Building on LabanLite, we present LaMoGen, a Text-to-LabanLite-to-Motion Generation framework that enables large language models (LLMs) to compose motion sequences through symbolic reasoning. The LLM interprets motion patterns, relates them to textual descriptions, and recombines symbols into executable plans, producing motions that are both interpretable and linguistically grounded. To support rigorous evaluation, we introduce a Labanotation-based benchmark with structured description–motion pairs and three metrics that jointly measure text–motion alignment across symbolic, temporal, and harmony dimensions. Experiments demonstrate that LaMoGen establishes a new baseline for both interpretability and controllability, outperforming prior methods on our benchmark and two public datasets. These results highlight the advantages of symbolic reasoning and agent-based design for language-driven motion synthesis. 
</details>


------------------------

## TODO List

- [x] Provide a project page
- [x] Release data pre-processing code
- [x] Release model and dataloader code
- [x] Release checkpoints and training/testing data
- [ ] Release the LLM data

------------------------

## Environment Setup

### Dependencies

The code can work on Windows with

```
pytorch                   2.1.2
pytorch-lightning         2.1.2
torchvision               0.16.2
CUDA                      11.8
```

Clone this repo and install the dependencies using the following script.

```python
conda create -n lamogen python=3.10
conda activate lamogen
# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
# install the rest of the dependencies
pip install -r requirements.txt
```

### Data Preparation

Detailed data preparation instructions are available in [scripts/README.md](scripts/README.md), which covers the full preprocessing pipeline from raw motion capture data to LabanLite symbol sequences. 
For convenience, we also provide pre-processed data bundles ready for training and evaluation.

------------------------

## Training and Inference

LaMoGen consists of two independently trained components:
1. **Laban Codec** (Decoder): Reconstructs motion features from discrete Laban symbol sequences.
2. **Laban Generator** (Code Generator): Predicts Laban symbol sequences conditioned on text embeddings.

### Training

#### 1. Train the Laban Codec

To train the codec on the HumanML3D dataset:

```bash
python -m train.train_codec --cfg codec/hml3d.yaml
```

All configuration files are located under the [`cfgs/`](cfgs) directory. You can modify the YAML files to adjust hyperparameters, dataset paths, or training settings.

#### 2. Train the Laban Generator

Once the codec is trained, train the generator with:

```bash
python -m train.train_code_gen --cfg code_gen/hml3d.yaml
```

### Inference

Pre-trained checkpoints are available [here](https://drive.google.com/file/d/1_m2u3Dx1_XztYI5LUHlqW2aq_DReeTfc/view?usp=sharing). Download and extract them to the project root, which will create `exp_codec/` and `exp_code_gen/` directories containing the decoder and generator weights respectively.

To generate motion sequences from pre-composed Laban symbol sequences:

```bash
python -m sample.t2m_llm_compose --cfg eval/t2m_llm_compose.yaml
```

Generated motion features will be saved as `.npy` files in the corresponding checkpoint directory (e.g., `exp_code_gen/exp_pae_0710_t2c_hml3d_ds1_fid_mask03/version_0/checkpoints/ckpt-epoch=0742-val_fid=0.23_t2c/`).

The LLM used for Laban composition is **Qwen3-8B**. You can compose your own Laban symbol sequences using the provided utility script [scripts\play_with_llm.py](scripts\play_with_llm.py).

------------------------

## Special Thanks

We would like to thank Ms. Wendy Chu Mang-Ching from the School of Dance, The Hong Kong Academy for Performing Arts, for the training and discussions on Labanotation.

## Acknowledgments

- **[SMPL/SMPL-X](https://smpl.is.tue.mpg.de/)**: For human body modeling
- **[BABEL-TEACH Dataset](https://github.com/atnikos/teach)**: For motion-text paired data
- **[HumanML3D](https://github.com/EricGuo5513/HumanML3D)**: For motion-text paired data, data processing, and text-motion evaluation
- **[priorMDM](https://github.com/priorMDM/priorMDM)**: For data processing and evaluators
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/pytorch3d/transforms/rotation_conversions.py)**: For rotation conversion utilities
- **[LabanotationSuite](https://github.com/microsoft/LabanotationSuite)**: For the initial idea and basic code
- **[TRAE Work](https://www.trae.ai/)**: For polishing/cleaning my redundant code with structure

## Citation

If you find this work helpful in your research, please consider leaving a star and citing:

```bibtex
@inproceedings{jiang2026lamogen,
  title={LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference},
  author={Jiang, Junkun and Au, Ho Yin and Xiang, Jingyu and Chen, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9364--9373},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
