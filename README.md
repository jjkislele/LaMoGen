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
- [] Release data pre-processing code
- [] Release model and dataloader code
- [] Release checkpoints and training/testing data

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

## Special Thanks

We would like to thank Ms. Wendy Chu Mang-Ching from the School of Dance, The Hong Kong Academy for Performing Arts, for the training and discussions on Labanotation.

## Acknowledgments

- **[SMPL/SMPL-X](https://smpl.is.tue.mpg.de/)**: For human body modeling
- **[BABEL-TEACH Dataset](https://github.com/atnikos/teach)**: For motion-text paired data
- **[HumanML3D](https://github.com/EricGuo5513/HumanML3D)**: For motion-text paired data, data processing, and text-motion evaluation
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/pytorch3d/transforms/rotation_conversions.py)**: For rotation conversion utilities
- **[LabanotationSuite](https://github.com/microsoft/LabanotationSuite)**: For the initial idea and basic code

## Citation

If you find this work helpful in your research, please consider leaving a star and citing:

```bibtex
@inproceedings{jiang2026lamogen,
  title={LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference},
  author={Jiang, Junkun and Au, Ho Yin and Xiang, Jingyu and Chen, Jie},
  booktitle={2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1--1},
  year={2026},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
