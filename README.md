# Corona face mask detection
→ In this project pre-trained Single Shot MultiBox Detector (SSD) face detector is used to detect all the faces in an input image and then our custom trained model is used to predict that the face contain a mask or not.

→ CNN model to predict a face contains a facemask or not is designed using the concepts of InceptionNet and ResNet. Basically we combined two inception blocks with two resent blocks. The architecture of our Convolutional Neural Network can be seen below: 

![](https://raw.githubusercontent.com/neilsaxena/Face-Mask-Prediction/main/Prediction/Model_Architecture.png)
→ After training it gives us an accuracy of 97.78%

→ “Face Mask ~12K Images Dataset” from kaggle is used for training the CNN.

→ All the training related stuff (like training jupyter notebook, keras tuner, dataset, etc) can be found in [Model Creation](https://github.com/neilsaxena/Face-Mask-Prediction/tree/main/Model%20Creation "Model Creation") folder.

→ For prediction there are python three programs:
 
 -  prediction.py (to run on live camera)
 - prediction_image.py (to run on image)
 - prediction_video.py (to run on video)

→ Prediction for an input image looks like this:

Input image: 

![](https://raw.githubusercontent.com/neilsaxena/Face-Mask-Prediction/main/Prediction/input_image.jpg)
Output Image:
![](https://raw.githubusercontent.com/neilsaxena/Face-Mask-Prediction/main/Prediction/image_output.jpg)
# Installation
→ In the command prompt first go to the directory in which requirements.txt file is present and then install all the required libraries using below command
```sh
pip  install  -r  requirements.txt
```
# Usage
→ In the command prompt navigate into prediction folder and then run the prediction file using python. For example use following command:

```sh
python prediction.py
```
