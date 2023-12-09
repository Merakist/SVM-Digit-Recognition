find_package(OpenCV REQUIRED)
add_executable(train train.cpp mnist_reader.h mnist_reader.cpp)
target_link_libraries(train ${OpenCV_LIBS})