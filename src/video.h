//
// Created by kai on 18-1-23.
//

#ifndef DARKNET_VIDEO_H


#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

typedef void* VideoCapture_T;

EXTERNC VideoCapture_T video_capture_init(char *url);
EXTERNC void video_capture_destroy(VideoCapture_T cap);
EXTERNC void *video_capture_read(VideoCapture_T cap);

#undef EXTERNC
// ...

#define DARKNET_VIDEO_H

#endif //DARKNET_VIDEO_H
