#include<iostream>
#include<filesystem>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::__fs::filesystem;

int main()
{
	string bmp_file;
	for (const auto& entry : fs::directory_iterator("../../data")) {
		if (entry.path().extension() == ".bmp") {
			bmp_file = entry.path().string();
			break;
		}
	}

	if (bmp_file.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

    Mat image = imread(bmp_file, 0);
    Mat img_show = image.clone();
    image.convertTo(image, CV_32F);
    image = image / 255.0;
    image = image.reshape(1, 1);

    // Attempt to Load SVM Model
    Ptr<SVM> svm;
    try {
        svm = StatModel::load<SVM>("../train/mnist_svm.xml");
    } catch (const cv::Exception& e) {
        cerr << "Error loading model: " << e.what() << endl;
        return -1;
    }

    // Check if the model is loaded successfully
    if (svm.empty()) {
        cout << "Failed to load mnist_svm.xml" << endl;
        return -1;
    }
   
    float ret = svm->predict(image);
    cout << "Prediction: " << ret << endl;

    return 0;
}
