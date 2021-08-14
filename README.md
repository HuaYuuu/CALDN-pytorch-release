# CALDN-pytorch-release
An official implementation of "Category-Aware Aircraft Landmark Detection" (2020 SPL)

### News
1. We have released the proposed PVALD dataset and the trained weight of CALDN.
2. The proposed PVALD dataset can be download via https://pan.baidu.com/s/1MUXIPrBGvwjF2hV9T1Hg2A(Code: i5ur) 
3. The trained weights of the proposed CALDN can be download via https://pan.baidu.com/s/1PnIzLXEvRb8Abgdf_BjT0g(Code: 3pin)

### Preparation
1. Download the Code of CALDN.
2. Follow the Installation of HRNet https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
3. Download images from FGVC dataset https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
4. Download the proposed PVALD dataset and the trained weight.

### Quick Start
1. Put images from FGVC dataset into './data/FGVC/data/images'
2. Put the trained weights into './trained_weight'
3. Run 'python tools/CALDN_test.py --cfg experiments/fgvc/CALDN/all_class/w32_256x256.yaml   TEST.MODEL_FILE trained_weight/landmark_final.pth CLASS_MODEL.PRETRAINED trained_weight/class_size_final.pth   CLASS_MODEL.PRETRAINED2 trained_weight/class_wing_final.pth   CLASS_MODEL.PRETRAINED3 trained_weight/class_tail_final.pth'

### PVALD Annotation Structure

First, the "Class_keypoints_info_train.npy" and the "Class_keypoints_info_test.npy" are needed to be downloaded.

Then, the meaning of each individual elements is:
 - `keypoint`  the landmark annotations of the corresponding image. "x" and "y" represent the coordinates of landmark. "visible" represents the visibility of each landmarks ("1" visible, "0" invisible caused by self-occlusion). "outside" represents if landmark locate outside of the image ("0" locate inside, "1" locate outside of the images. Note that if landmark locate outside the image, the corresponding coordinates represent the nearest point in the images to the true landmark location.)
 - `box`  the bounding box of aircraft in the image.
 - `split` the split, i.e. `train`/`test`.
 - `class`   the category annotation of each aircraft instance. Each aircraft instance possesses three category label, corresponding to the classification of size, wing and tail. "Size" category including large("1")/medium("2")/small("3"). "Wing" category representing the wings of aircraft locate above the fuselage("1") or under the fuselage("2"). "Tail" category representing the horizontal stabilizer of aircraft locate above the vertical stabilizer("1"), in the middle of the vertical stabilizer("2"), or under the vertical stabilizer("3").

### Citation
If you use PVALD dataset or the code of CALDN in your work, please cite our publication as listed.

@ARTICLE{CALDN,
  author={Li, Yi and Chang, Yi and Ye, Yuntong and Zou, Xu and Zhong, Sheng and Yan, Luxin},
  journal={IEEE Signal Processing Letters}, 
  title={Category-Aware Aircraft Landmark Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={61-65},
  doi={10.1109/LSP.2020.3045623}}

@ARTICLE{SALD,
  author={Ye, Yuntong, Chang, Yi and Li, Yi and Yan, Luxin},
  journal={International Conference on Image and Graphics 2021}, 
  title={Skeleton-Aware Network for Aircraft Landmark Detection}, 
  year={2021}}

### Contact

Yi Li

li_yi@hust.edu.cn
