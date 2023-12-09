#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "mnist_reader.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
	// Load training data
    string train_images_path = "../../data/train-images.idx3-ubyte";
    string train_labels_path = "../../data/train-labels.idx1-ubyte";
    Mat train_labels = read_mnist_label(train_labels_path);
    Mat train_images = read_mnist_image(train_images_path);

    // Check if train_labels is empty
    if (train_labels.empty()) {
        cerr << "Error: train_labels is empty. Unable to read from " << train_labels_path << endl;
        return -1;
    }

    // Check if train_images is empty
    if (train_images.empty()) {
        cerr << "Error: train_images is empty. Unable to read from " << train_images_path << endl;
        return -1;
    }
    train_images = train_images / 255.0;  // Normalize

    // Create and train SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY);
    svm->setGamma(3.0);
    svm->setDegree(3.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 300, 0.0001));

    Ptr<TrainData> train_data = TrainData::create(train_images, ROW_SAMPLE, train_labels);
    cout << "Training SVM..." << endl;
    svm->train(train_data);
    cout << "Training completed" << endl;

    // Load test data
    string test_images_path = "../../data/t10k-images.idx3-ubyte";
    string test_labels_path = "../../data/t10k-labels.idx1-ubyte";
    Mat test_labels = read_mnist_label(test_labels_path);
    Mat test_images = read_mnist_image(test_images_path);

    if (test_labels.empty()) {
        cerr << "Error: test_labels is empty. Unable to read from " << test_labels_path << endl;
        return -1;
    }

    // Check if train_images is empty
    if (test_images.empty()) {
        cerr << "Error: test_images is empty. Unable to read from " << test_images_path << endl;
        return -1;
    }
    test_images = test_images / 255.0;  // Normalize

    // Predict using SVM
    Mat pre_out;
    cout << "Predicting..." << endl;
    svm->predict(test_images, pre_out);
    cout << "Prediction completed" << endl;

    // Calculate accuracy
    pre_out.convertTo(pre_out, CV_8UC1);
    test_labels.convertTo(test_labels, CV_8UC1);

    int equal_nums = 0;
    for (int i = 0; i < pre_out.rows; i++) {
        if (pre_out.at<uchar>(i, 0) == test_labels.at<uchar>(i, 0)) {
            equal_nums++;
        }
    }
    float acc = float(equal_nums) / float(pre_out.rows);
    cout << "Accuracy on test dataset: " << acc * 100 << "%" << endl;

    // Save the trained model
    svm->save("mnist_svm.xml");

    return 0;
}
