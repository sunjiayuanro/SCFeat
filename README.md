# Shared Coupling-bridge for Weakly Supervised Local Feature Learning

This repository contains a PyTorch implementation of the paper:

<!-- [*Shared Coupling-bridge for Weakly Supervised Local Feature Learning*](https://sunjiayuanro.github.io/SCFeat/) -->
<!-- [[Project page]](https://sunjiayuanro.github.io/SCFeat/) -->
[[Arxiv]](https://arxiv.org/abs/2212.07047)

<!-- [Jiayuan Sun](),  -->
<!-- [Luping Ji](), -->
<!-- [Jiewen Zhu]()  -->
Jiayuan Sun,
Luping Ji,
Jiewen Zhu

<!-- TMM -->

## Abstract

Sparse local feature extraction is usually believed to be of important significance in typical vision tasks such as simultaneous localization and mapping, image matching and 3D reconstruction. At present, it still has some deficiencies needing further improvement, mainly including the discrimination power of extracted local descriptors, the localization accuracy of detected keypoints, and the efficiency of local feature learning. This paper focuses on promoting the currently popular sparse local feature learning with camera pose supervision. Therefore, it pertinently proposes a Shared Coupling-bridge scheme with four light-weight yet effective improvements for weakly-supervised local feature (SCFeat) learning. It mainly contains: i) the Feature-Fusion-ResUNet Backbone (F2R-Backbone) for local descriptors learning, ii) a shared coupling-bridge normalization to improve the decoupling training of description network and detection network, iii) an improved detection network with peakiness measurement to detect keypoints and iv) the fundamental matrix error as a reward factor to further optimize feature detection training. Extensive experiments prove that our SCFeat improvement is effective. It could often obtain a state-of-the-art performance on classic image matching and visual localization. In terms of 3D reconstruction, it could still achieve competitive results.


## Requirements
```bash
# Create conda environment with torch 1.8.2 and CUDA 11.1
conda env create -f environment.yml
conda activate scfeat
```
If you encounter problems with OpenCV, try to uninstall your current opencv packages and reinstall them again
```bash
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-python==3.4.2.17
pip install opencv-contrib-python==3.4.2.17
```

## Pretrained Model
<!-- Pretrained model can be downloaded using this google drive [link]() -->
TODO

## Training
Download the preprocessed subset of MegaDepth from [CAPS](https://github.com/qianqianwang68/caps), and run the following command: 

```bash
# train the description network
python train.py --config config/train_desc.yaml
```

```bash
# example train the detection network
python train.py --config config/train_det.yaml
```

## Feature extraction
We provide code for extracting SCFeat features on HPatches dataset.
To download and use the HPatches Sequences, please refer to this [link](https://github.com/mihaidusmanu/d2-net/tree/master/hpatches_sequences).

To extract SCFeat features on HPatches dataset, download the pretrained model, modify paths in ```configs/extract_hpatches.yaml``` and run
```bash
python extract_features.py --config config/extract_hpatches.yaml
```

TODO

## BibTeX
If you use this code in your project, please cite the following paper:
```bibtex
@article{DBLP:journals/corr/abs-2212-07047,
  author    = {Jiayuan Sun and
               Jiewen Zhu and
               Luping Ji},
  title     = {Shared Coupling-bridge for Weakly Supervised Local Feature Learning},
  journal   = {CoRR},
  volume    = {abs/2212.07047},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2212.07047},
  doi       = {10.48550/arXiv.2212.07047},
  eprinttype = {arXiv},
  eprint    = {2212.07047},
  timestamp = {Mon, 02 Jan 2023 15:09:55 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2212-07047.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

