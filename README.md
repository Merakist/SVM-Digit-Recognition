# SVM-based Digit Recognition

A SVM-based OpenCV Project for recognizing handwritten digits in C++.

This project uses the [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/).

## Installation

1. Visit the [official OpenCV website](https://opencv.org/get-started/) and install OpenCV.

2. Clone this Repository (SVM-Recognition).

3. Download MNIST Data and put the data files under `SVM-Recognition/data`.

> ```bash
> t10k-images.idx3-ubyte
> t10k-labels.idx1-ubyte
> train-images.idx3-ubyte
> train-labels.idx1-ubyte
> ```

4. Input the following command in your command prompt to run.

**NOTE:** You need to run every command of this recipe in the `SVM-Recognition` root path:
> ```bash
> cd build
> cmake ..
> ```

## Usage

### Train

To train a SVM model, run the following command.

**NOTE:** You need to run every command of this recipe in the `SVM-Recognition` root path:
> ```bash
> cd build
> make train
> cd train
> ./train
> ```

The program will train a SVM model by the training data `train-images.idx3-ubyte` and `train-labels.idx1-ubyte`, and will calculate the accuracy by `t10k-images.idx3-ubyte` and `t10k-labels.idx1-ubyte`.


### Recognition

To run predictions of an image of a digit, make sure the image conforms to the format in MNIST database, i.e., a 28*28 bmp image. The filename of the image does not matter. So long as the image is in the right format and is a .bmp file, the program will run a prediction. 

- **Extracted MNIST Image by 3omar-mostafa**: [Github Repo](https://github.com/3omar-mostafa/MNIST-dataset-extractor)

To start, put the image under `SVM-Recognition/data`. The program will only run predictions for the first occurrence of a .bmp file, so don't put more than one image.

**NOTE:** You need to run every command of this recipe in the `SVM-Recognition` root path:
> ```bash
> cd build
> make recognition
> cd recognition
> ./recognition
> ```

The program will then print the prediction of the image.


## Pre-trained Model

A pre-trained model is included: `/SVM-Recognition/build/mnist_svm.xml` with accuracy 97.54%.

## Related Reading in Digit Recognition

- **K Nearest Neighbors (KNN)**: [Fundamentals](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/), [OpenCV Example](https://docs.opencv.org/4.x/dd/de1/classcv_1_1ml_1_1KNearest.html)

- **Support Vector Machine (SVM)**: [Fundamentals](https://www.kaggle.com/code/prashant111/svm-classifier-tutorial), [OpenCV Example](https://docs.opencv.org/3.4/d1/d73/tutorial_introduction_to_svm.html)

- **K-means clustering (Kmeans)**: [Fundamentals](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/), [OpenCV Example](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html)

- **Decision Trees**: [Fundamentals](https://www.geeksforgeeks.org/decision-tree/), [OpenCV Example](https://docs.opencv.org/4.x/dc/dd6/ml_intro.html)

- **Deep Neural Network (DNN)**: [Fundamentals](https://www.simplilearn.com/tutorials/deep-learning-tutorial/multilayer-perceptron), [OpenCV Example](https://docs.opencv.org/4.x/dc/dd6/ml_intro.html)
