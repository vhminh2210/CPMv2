# Experimenting Color-Pattern Makeup Transfer with UV-Position Transformation

This repository explores advanced techniques in makeup transfer, building on the foundational work by [VinAIResearch/CPM](https://github.com/VinAIResearch/CPM). Our goal is to enhance the color and pattern transfer capabilities by incorporating a UV-position transformation for pose adjustment. This project serves both as a reproduction and extension of the CPM codebase to push the boundaries of makeup transfer applications.

## Project Overview

In this project, we:

1. **Reproduce, and debug** the official repository of the paper *"Lipstick Ain't Enough: Beyond Color Matching for In-the-Wild Makeup Transfer"* by Thao Nguyen, Anh Tran, and Minh Hoai (CVPR 2021). This repository can be found at [VinAIResearch/CPM](https://github.com/VinAIResearch/CPM).

```bibtex
@inproceedings{m_Nguyen-etal-CVPR21,
  author = {Thao Nguyen and Anh Tran and Minh Hoai},
  title = {Lipstick ain't enough: Beyond Color Matching for In-the-Wild Makeup Transfer},
  year = {2021},
  booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```

**Please CITE** this paper if you use this dataset or model implementation in your own research or software.

2. **Propose a pose-transformation learning framework** for makeup transfer tasks. By leveraging UV-position mapping, we aim to improve makeup consistency across varied face angles and lighting. This work is still under active investigation, with ongoing efforts to stabilize and enhance performance.

## Getting Started

### Prerequisites

1. Python 3.8+
2. Dependencies listed in `requirements.txt`
3. Follow environment setups described in the original repository [VinAIResearch/CPM](https://github.com/VinAIResearch/CPM), regarding the use of dataset and pretrained models.

To install the necessary packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Reproducing Original CPM Model** :

   Follow the instructions in the `CPM_v2.ipynb` notebook to reproduce the original results from the CPM repository. Library warnings can be ignored.

2. **Pose-Transformation Makeup Transfer** :

   Our modified pipeline for UV-position transformation can be found at `CPM_v2plus.ipynb` notebook.

### Results and Discussion

- **Original CPM Reproduction**: The original CPM model has been successfully reproduced with library-related bugs fixed.
- **UV-Position Transformation**: Preliminary experiments show that the method needs to be studied more to bring improvements to makeup consistency across varied poses. Further investigations are needed to refine the performance and address limitations.

## Acknowledgments

This project is based on the work from the paper _"Lipstick Ain't Enough: Beyond Color Matching for In-the-Wild Makeup Transfer"_ by Thao Nguyen, Anh Tran, and Minh Hoai. Special thanks to the authors and the VinAIResearch team for their valuable contributions.

```bibtex
@inproceedings{m_Nguyen-etal-CVPR21,
  author = {Thao Nguyen and Anh Tran and Minh Hoai},
  title = {Lipstick ain't enough: Beyond Color Matching for In-the-Wild Makeup Transfer},
  year = {2021},
  booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```

## License

This project follows the license terms of the original CPM repository.
