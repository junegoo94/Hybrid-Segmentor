## 	Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructure

## 1. Model Architecture
![](./figures/model_architecture.png)

You can download Hybrid-Segmentor model weights from [this link](https://1drv.ms/u/s!AtFigR8so_Ssr74TeEfNbT0255DU3w?e=yq2xul)
#### If you use our model in your research, please cite "Hybrid-Segmentor Reference" below.

## 2. Refined Dataset (CrackVision12K)
The refined dataset is developed with 13 publicly available datasets that have been refined using image processing techniques.
**Please note that the use of our dataset is RESTRICTED to non-commercial research and educational purposes.**

You can download the dataset from [this link](https://1drv.ms/u/s!AtFigR8so_Ssr74aBKrrJ0ifkWLZpg?e=57fmSb).
|Folder|Sub-Folder|Description|
|:----|:-----|:-----|
|`train`|IMG / GT|RGB images and binary annotation for training|
|`test`|IMG / GT|RGB images and binary annotation for testing|
|`val`|IMG / GT|RGB images and binary annotation for validation|

#### To download the dataset from the link, please cite "Dataset Reference" below.

## 3. Set-Up
**Training**
Before training, change variables such as dataset path, batch size, etc in config.py. 
```
python trainer.py
```

**Testing**
Before testing, change the model name and output folder path.
```
python test.py
```
## Citaiton
 - **Hybrid-Segmentor & CrackVision12K Reference**:
   If you use our model or dataset, please cite the following:
```
@misc{goo2024hybridsegmentorhybridapproachautomated,
      title={Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructure}, 
      author={June Moh Goo and Xenios Milidonis and Alessandro Artusi and Jan Boehm and Carlo Ciliberto},
      year={2024},
      eprint={2409.02866},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02866}, 
}
```
 - **Dataset Reference**:
1. Aigle-RN / ESAR / LCMS Datasets [Dataset Link](https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html)
```
@article{AEL_dataset,
  title={Automatic crack detection on two-dimensional pavement images: An algorithm based on minimal path selection},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={10},
  pages={2718--2729},
  year={2016},
  publisher={IEEE}
}
```
2. SDNet2018 Datasets [Dataset Link](https://digitalcommons.usu.edu/all_datasets/48/)
```
@article{sdnet2018,
  title={SDNET2018: A concrete crack image dataset for machine learning applications},
  author={Maguire, Marc and Dorafshan, Sattar and Thomas, Robert J},
  year={2018},
  publisher={Utah State University}
}
```
3. Masonry Datasets [Dataset Link](https://github.com/dimitrisdais/crack_detection_CNN_masonry)
```
@article{masonry_dataset,
  author = {Dais, Dimitris and Bal, Ihsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},
  doi = {10.1016/j.autcon.2021.103606},
  journal = {Automation in Construction},
  pages = {103606},
  title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},
  volume = {125},
  year = {2021}
}
```
4. Crack500 Dataset [Dataset Link](https://github.com/fyangneil/pavement-crack-detection)
```
@inproceedings{crack500_dataset,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={2016 IEEE international conference on image processing (ICIP)},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}
```
5. CrackLS315 / CRKWH100 / CrackTree260 / Stone331 Datasets [Github Link](https://github.com/qinnzou/DeepCrack) [Direct Link-passcodes: zfoo](https://pan.baidu.com/s/1PWiBzoJlc8qC8ffZu2Vb8w)
```
@article{Deep_crack_crackLS315,
  title={Deepcrack: Learning Hierarchical Convolutional Features for Crack Detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}
```
6. DeepCrack Dataset [Dataset Link](https://github.com/yhlleo/DeepCrack)
```
@article{deepcrack_dataset,
title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
journal={Neurocomputing},
volume={338},
pages={139--153},
year={2019},
doi={10.1016/j.neucom.2019.01.036}
}
```
7.1 GAPS384 7.2 GAPs (Original Dataset and paper) [GAPS384 Dataset Link](https://github.com/fyangneil/pavement-crack-detection) [GAPs Dataset Link](https://www.tu-ilmenau.de/neurob/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)
```
@article{FPHBN_gaps384,
title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
journal={IEEE Transactions on Intelligent Transportation Systems}, year={2019}, publisher={IEEE} }

@inproceedings{GAPS_data_original,
title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike and Gross, Horst-Michael},
booktitle={International Joint Conference on Neural Networks (IJCNN)}, pages={2039--2047}, year={2017} }
```
8. CFD Dataset [Dataset Link](https://github.com/cuilimeng/CrackForest-dataset)
```
@article{CFD1,
title={Automatic road crack detection using random structured forests},
author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
journal={IEEE Transactions on Intelligent Transportation Systems},volume={17},number={12},
pages={3434--3445},year={2016},publisher={IEEE}}

@inproceedings{CFD2,
title={Pavement Distress Detection Using Random Decision Forests},
author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},
booktitle={International Conference on Data Science},
pages={95--102},
year={2015},
organization={Springer}
}
```

If you have any questions, please contact me: june.goo.21 @ ucl.ac.uk without hesitation.
