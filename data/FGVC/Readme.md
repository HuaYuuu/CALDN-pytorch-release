# The Perspective-Variant Aircraft Landmark Dataset

The Perspective-Variant Aircraft Landmark Dataset derives from the FGVC dataset and constitutes a collection of partially annotations about aircraft's landmarks along with the corresponding category information.

Due to licensing issues, the images from FGVC dataset is only available for download at the website of the original FGVC dataset [FGVC].


In the following, we provide details on the annotation about the landmarks of aircraft and the corresponding category label.

### Annotation Structure

First, the "Class_keypoints_info_train.npy" and the "Class_keypoints_info_test.npy" are needed to be downloaded.

Then, the meaning of each individual elements is:
 - `keypoint`  the landmark annotations of the corresponding image. "x" and "y" represent the coordinates of landmark. "visible" represents the visibility of each landmarks ("1" visible, "0" invisible caused by self-occlusion). "outside" represents if landmark locate outside of the image ("0" locate inside, "1" locate outside of the images. Note that if landmark locate outside the image, the corresponding coordinates represent the nearest point in the images to the true landmark location.)
 - `box`  the bounding box of aircraft in the image.
 - `split` the split, i.e. `train`/`test`.
 - `class`   the category annotation of each aircraft instance. Each aircraft instance possesses three category label, corresponding to the classification of size, wing and tail. "Size" category including large("1")/medium("2")/small("3"). "Wing" category representing the wings of aircraft locate above the fuselage("1") or under the fuselage("2"). "Tail" category representing the horizontal stabilizer of aircraft locate above the vertical stabilizer("1"), in the middle of the vertical stabilizer("2"), or under the vertical stabilizer("3").

### Citation

If you use *Perspective-Variant Aircraft Landmark Dataset* in your work, please cite our publication as listed and the FGVC dataset [FGVC].

@ARTICLE{9298853,
  author={Li, Yi and Chang, Yi and Ye, Yuntong and Zou, Xu and Zhong, Sheng and Yan, Luxin},
  journal={IEEE Signal Processing Letters}, 
  title={Category-Aware Aircraft Landmark Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={61-65},
  doi={10.1109/LSP.2020.3045623}}

### Contact

Yi Li

li_yi@hust.edu.cn


[FGVC]: <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>