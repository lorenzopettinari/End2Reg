# End2Reg: Learning Task-Specific Segmentation for Markerless Registration in Spine Surgery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue)](https://lorenzopettinari.github.io/end-2-reg/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.13402-B31B1B.svg)](https://arxiv.org/abs/2512.13402)


This repository contains the Pytorch implementation of "End2Reg: Learning Task-Specific Segmentation for Markerless Registration in Spine Surgery".

This repository provides an optimized implementation of **End2Reg**. Compared to the original paper where the reported inference latency was **640 ms** (1.5 FPS), this codebase is significantly optimized to achieve faster inference (**180 ms** / >5.3 FPS).



## Installation & Setup

### System Prerequisites
- **Python**: `>= 3.10` (tested on `3.10`, `3.11`, `3.12`)
- **CUDA**: `>= 11.8` (tested up to `13.x`)

### 1. Create a Conda Environment

```bash
conda create -n myenv python=3.11
conda activate myenv
```

### 2. Install PyTorch with CUDA support
- **Pytorch**: `>= 2.2.0`

### 3. Install Repository & Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```


## Project Structure

- `end2reg/`: Core library package.
  - `ops/`: Pure PyTorch implementations of spatial point cloud operations (`knn`, `grid_subsample`, etc.).
  - `datasets/`: Dataset definitions for SpineDepth.
  - `modules/`: Loss functions, sinkhorn matching, transformers, and registration layers.
- `experiments/`: Experiment configurations and trainer definitions.
  - `end2reg.spinedepth/`: End2Reg pipeline for SpineDepth registration.



## Dataset Structure

The SpineDepth registration dataset is expected to be organized into train and test splits, indexed via metadata Excel files:

```text
dataset_root/
├── Dataset_info_train.xlsx
├── Dataset_info_test.xlsx
├── Train/
│   └── Spec_<spec_id>/
│       └── recording<rec_id>/
│           ├── Pointclouds_<vid_id>/
│           │   └── point_cloud_frame_<frame_id>.ply   # Reference intraoperative point cloud (positions, colors, labels)
│           └── Vertebrae_<vid_id>/
│               └── vertebrae_frame_<frame_id>.ply     # Source preoperative vertebra point cloud
└── Test/
    └── Spec_<spec_id>/
        └── recording<rec_id>/
            ├── Pointclouds_<vid_id>/
            │   └── point_cloud_frame_<frame_id>.ply
            └── Vertebrae_<vid_id>/
                └── vertebrae_frame_<frame_id>.ply
```

### Data Annotations
- **Reference Point Cloud (`Pointclouds_*/point_cloud_frame_*.ply`)**: Acquired intraoperative RGB-D point cloud containing point 3D coordinates (`positions`), RGB channels (`colors`), and binary weak labels segmentation used for comparison (`labels`: `0` for background tissue, `1` for bone/vertebra). **No segmentation label is used at training or inference time by our model.**
- **Source Point Cloud (`Vertebrae_*/vertebrae_frame_*.ply`)**: Preoperative 3D vertebrae point cloud to be registered.
- **Metadata (`Dataset_info_{train,test}.xlsx`)**: Contains index columns (`spec_num`, `rec_num`, `vid_num`, `frame_num`) along with the flattened 4x4 ground truth rigid transformation matrix (`GT_transformation_matrix`).



## Training & Evaluation

### Training
To train the End2Reg model from scratch on the SpineDepth dataset:
```bash
python experiments/end2reg.spinedepth/trainval.py --dataset_dir /path/to/dataset
```

### Evaluation
To evaluate a trained checkpoint snapshot on the test set:
```bash
python experiments/end2reg.spinedepth/test.py --dataset_dir /path/to/dataset --test_epoch <iteration_number>
```



## Citation

If you find this work or code useful in your research, please cite:

```bibtex
@inproceedings{pettinari2026end2reg,
  title={End2Reg: Learning Task-Specific Segmentation for Markerless Registration in Spine Surgery},
  author={Pettinari, Lorenzo and El Hadramy, Sidaty and Wehrli, Michael and Cattin, Philippe C and Studer, Daniel and Hasler, Carol C and Licci, Maria},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={66--76},
  year={2026},
  organization={Springer}
}
```



## License

This project is released under the [MIT License](LICENSE).



## Acknowledgements

This codebase builds upon and is inspired by the following open-source repositories:
- [GeoTransformer](https://github.com/qinzheng93/geotransformer): Geometric Transformer for Fast and Robust Point Cloud Registration.
- [Easy-KPConv](https://github.com/qinzheng93/Easy-KPConv): PyTorch-friendly implementation of Kernel Point Convolution (KPConv).

