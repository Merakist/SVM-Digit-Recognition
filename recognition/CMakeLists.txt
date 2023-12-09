find_package(OpenCV REQUIRED)
add_executable(recognition recognition.cpp)
target_link_libraries(recognition ${OpenCV_LIBS})