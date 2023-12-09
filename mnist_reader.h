// mnist_reader.h
#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <opencv2/opencv.hpp>
#include <string>

int reverseInt(int i);
cv::Mat read_mnist_image(const std::string fileName);
cv::Mat read_mnist_label(const std::string fileName);

#endif // MNIST_READER_H
