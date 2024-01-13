# Variable-Wise Diagonal Preconditioning for Primal-Dual Splitting: Design and Applications

This is a demo code of the method proposed in the following reference:

K. Naganuma and S. Ono
``Variable-Wise Diagonal Preconditioning for Primal-Dual Splitting: Design and Applications''

Update history:
Augast 7, 2023: v1.0 

For more information, see the following 
- Official: https://ieeexplore.ieee.org/document/10215352
- Project website: https://www.mdi.c.titech.ac.jp/publications/ovdp
- Preprint paper: https://arxiv.org/abs/2301.08468

# How to use
1. Download datasets from https://drive.google.com/file/d/1p7ePx0e2RUy3uzmnphEw77nrQrglrQcX/view?usp=sharing

2. Run demo codes.
 - For mixed noise removal, run demo_MNR.m
 - For unmixing, run demo_unmixing.m
 - For graph signal recovery, run demo_GSR.m

# Others
The following two files were downloaded at https://epfl-lts2.github.io/gspbox-html/ (GSPBOX)
 - ./GSR/utils/gsp_graph_default_plotting_parameters.m
 - ./GSR/utils/gsp_plot_signal.m

# Reference
If you use this code, please cite the following paper:

```
@ARTICLE{10215352,
  author={Naganuma, Kazuki and Ono, Shunsuke},
  journal={IEEE Transactions on Signal Processing}, 
  title={Variable-Wise Diagonal Preconditioning for Primal-Dual Splitting: Design and Applications}, 
  year={2023},
  volume={71},
  number={},
  pages={3281-3295},
  doi={10.1109/TSP.2023.3304789}
}
```
