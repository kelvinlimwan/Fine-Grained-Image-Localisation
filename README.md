# Final Project: Fine-Grained Localisation
This is for the final assignment for Computer Vision (COMP90086). Given a dataset of images with an associated label containing their location, we aim to investigate methods to recognize the location of the images. 

To this end, we benchmark various feature decriptors/extractors and image matching techniques.
<br/><b>Descriptors/Features:</b>
- SIFT
- ASIFT
- VladNET
- Self-supervised ANN (Rotation)
- Self-supervised ANN (Warp)

<b>Image Matching:</b>
- KNN
- FLaNN
- MLP

### Dependencies
This program was developed using:
1. Python 3.9.7
2. Keras 2.6.0
3. Tensorflow 2.6.0

<br/>
Pretrained weights for NetVLAD "netvlad_weights.h5" must be downloaded and put into the <b>checkpoint</b> directory before running. Download <a href="https://onedrive.live.com/?authkey=%21AM3LfsRZTJ1TOHI&cid=318792FBF3A5A7EB&id=318792FBF3A5A7EB%21290981&parId=318792FBF3A5A7EB%21290980&action=locate">here</a> 
(Obtained from Github <a href="https://github.com/crlz182/Netvlad-Keras">repository</a> "Netvlad-Keras")


### How to Run the Code
The main file is "Experiments.ipynb"
