# HO-3D - scripts

<div style="text-align:center"><img src="teaser.png" width="60%"></div>


# :exclamation: :exclamation: :exclamation: New:exclamation: :exclamation: :exclamation:
We have now released **version 3** of the HO-3D dataset (HO-3D_v3) with **more accurate hand-object poses**. See this [report](https://arxiv.org/abs/2107.00887) and official
 [website](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/) for details and links.
 The annotation format and folder structure follows almost the same format as the previous version (HO-3D_v2) and hence replacing the old dataset with
 the new one should work just fine. The only change being all the rgb images are now in 'jpg' format instead of 'png' format due to storage constraints.
 
 A new codalab challenge for **version 3** has been created [here](https://competitions.codalab.org/competitions/33267). Submission to this new challenge follows the exact same format as for
  [version 2](https://competitions.codalab.org/competitions/22485). The codalab challenge for [HO-3D_v2](https://competitions.codalab.org/competitions/22485) will **close** on Dec. 1, 2021
 <hr />
 


HO-3D is a dataset with 3D pose annotations for hand and object under severe occlusions from each other. The sequences in the dataset contain different
persons manipulating different objects, which are taken from [YCB dataset](https://rse-lab.cs.washington.edu/projects/posecnn/). Details about the 
proposed annotation method can be found in our [paper](https://arxiv.org/pdf/1907.01481). The dataset (HO-3D_v3) contains 103,462 annotated images and their 
corresponding depth maps.

For more details about the dataset and the corresponding work, visit our [project page](https://www.tugraz.at/index.php?id=40231)

An online codalab challenge which provides a platform to evaluate different hand pose estimation methods on our dataset with standard metrics is launched 
[here](https://competitions.codalab.org/competitions/22485) (for version 2) and [here](https://competitions.codalab.org/competitions/33267) (for version 3)

This repository contains a collection of scripts for:
* Visualization of HO-3D dataset
* Evaluation scripts used in the challenge


# Basic setup

1. Install basic requirements:
    ```
    conda create -n python2.7 python=2.7
    source activate python2.7
    pip install numpy matplotlib scikit-image transforms3d tqdm opencv-python cython open3d
    ```
2. Download Models&code from the MANO website
    ```
    http://mano.is.tue.mpg.de
    ```
3. Assuming ${MANO_PATH} contains the path to where you unpacked the downloaded archive, use the provided script to setup the MANO folder as required.
    ```
    python setup_mano.py ${MANO_PATH}
    ```
    
4. Download the YCB object models by clicking on `The YCB-Video 3D Models` in [https://rse-lab.cs.washington.edu/projects/posecnn/]. Assume ${YCB_PATH}
is the path where you unpacked the object models into (path to where _models_ folder branches off)

5. Download the HO-3D dataset. See project page for instructions. 
    
6. Assuming ${DB_PATH} is the path to where you unpacked the dataset (path to where _./train/_ and _./evaluation/_ folder branch off), 
This should enable you to run the following to show some dataset samples.
    ```
    python vis_HO3D.py ${DB_PATH} ${YCB_PATH}
    python vis_HO3D.py ${DB_PATH} ${YCB_PATH} -split 'evaluation'
    python vis_HO3D.py ${DB_PATH} ${YCB_PATH} -visType 'open3d' 
    ```
    
The script provides parameters to visualize the annotations in 3D using open3d or in 2D in matplotlib window. Use `-visType` to set the visualization type.
The script also provides parameters to visualize samples in the training and evaluation split using the parameters `-split`.


# Evaluate on the dataset

In order to have consistent evaluation of the hand pose estimation algorithms on HO-3D dataset, evaluation is handled through CodaLab competition.
 
1. Make predictions for the evaluation dataset. The code provided here predicts zeros for all joints and vertices. ${ver} specifies the version
of the dataset ('v2' or 'v3')
    ```
    python pred.py ${DB_PATH} --version ${ver}
    ```
     
2. Zip the `pred.json` file
    ```
    zip -j pred.zip pred.json
    ```
    
3. Upload `pred.zip` to our Codalab competition ([version 2](https://competitions.codalab.org/competitions/2dsd)
 or [version3](https://competitions.codalab.org/competitions/33267))website (Participate -> Submit)

4. Wait for the evaluation server to report back your results and publish your results to the leaderboard. The zero predictor will give you the following results
    ```
    Mean joint error 56.87cm
    Mean joint error (procrustes alignment) 5.19cm
    Mean joint error (scale and trans alignment) NaN
    Mesh error 57.12cm
    Mesh error (procrustes alignment) 5.47cm
    F@5mm=0.0, F@15mm=0.0
    F_aliged@5mm= 0.000, F_aligned@15mm=0.017
    ```
    
5. Modify `pred.py` to use your method for making hand pose estimation and see how well it performs compared to the baselines. Note that the pose
estimates need to be in **OpenGL** coordinate system (hand is along negative z-axis in a right-handed coordinate system with origin at camera optic center)
during the submission. 

6. The calculation of the evaluation metrics can be found in `eval.py`

# Visualize Point Cloud from All the Cameras (only in version 3)

We provide the extrinsic camera parameters in 'calibration' folder of the dataset. The RGB-D data from all the cameras 
for multi-camera sequences can be combined to visualize the point-cloud using the below script:
   ```python
    python vis_pcl_all_cameras.py ${DB_PATH} --seq SEQ --fid FID
```
`SEQ` and `FID` are the sequence name and file name. Try `-h` for list of accepted sequence names.

# Compare with Manual Annotations (only in version 3)

We manually annotated 5 finger tip locations in 53 frames using the point-cloud from all the cameras. The manually annotated
 finger tip locations are provided in 'manual_annotations' folder of the dataset. We measure the accuracy of our automatic 
annotations by comparing with the manual annotations using the below script:
```python
python compare_manual_anno.py ${DB_PATH}
```

 
# Terms of use

The download and use of the dataset is for academic research only and it is free to researchers from educational or research institutes
for non-commercial purposes. When downloading the dataset you agree to (unless with expressed permission of the authors): 
not redistribute, modificate, or commercial usage of this dataset in any way or form, either partially or entirely. 
If using this dataset, please cite the corresponding paper.

    
	@INPROCEEDINGS{hampali2020honnotate,
    	      title={HOnnotate: A method for 3D Annotation of Hand and Object Poses},
              author={Shreyas Hampali and Mahdi Rad and Markus Oberweger and Vincent Lepetit},
              booktitle = {CVPR},
	      year = {2020}
             }
    
# Acknowledgments

1. The evaluation scripts used in the HO-3D challenge have been mostly re-purposed from [Freihand challenge](https://github.com/lmb-freiburg/freihand). We
thank the authors for making their code public.

2. This work was supported by the Christian Doppler Laboratory for Semantic 3D Computer Vision, funded in part
by Qualcomm Inc
