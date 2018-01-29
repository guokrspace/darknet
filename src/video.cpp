//
// Created by kai on 18-1-23.
//
#include "video.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
/*
 * 1) Open Video Streaming
 * 2) Fetch Video Frame in Thread
 * 3) Predict the Video Frame in Thread
 */


using namespace std;
using namespace cv;

VideoCapture_T video_capture_init(char *url) {
    VideoCapture *capture = new VideoCapture(url);
    //     cv::VideoCapture capture(url);

     if (!capture->isOpened()) {
         //Error
         std::cout << "Error opening video stream or file" << std::endl;
         return NULL;
     }

     return capture;
}

void video_capture_destroy(VideoCapture_T cap) {
    VideoCapture* capture = static_cast<VideoCapture *>(cap);
    delete capture;
}

void *video_capture_read(VideoCapture_T cap) {
    cv::Mat frame;
    VideoCapture* capture = static_cast<VideoCapture *>(cap);
    capture->read(frame);
    IplImage *img;
    img = new IplImage(frame);
    return img;
}