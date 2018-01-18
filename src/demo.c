#include "darknet.h"
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1
#ifdef OPENCV

#define NUM_VIDEO_CHANNEL 2

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

float **probs;
box *boxes;
static network *net;
image buff [NUM_VIDEO_CHANNEL][3];
image buff_letter[NUM_VIDEO_CHANNEL][3];
int buff_index[NUM_VIDEO_CHANNEL];
static CvCapture * cap[NUM_VIDEO_CHANNEL];
IplImage  * ipl[NUM_VIDEO_CHANNEL];
char window_name[NUM_VIDEO_CHANNEL][16];
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *avg;
double demo_time;

static log4c_category_t* logger = NULL;

redisContext *c;
redisReply *reply;

char output_buf[1024];

typedef struct thread_arg{
    char *window_name;
    image buff [3];
    image buff_letter[3];
    int buff_index;
    CvCapture * cap;
    IplImage  * ipl;
}T_Thread_Arg;

void *detect_in_thread(void *ptr)
{
    T_Thread_Arg *arg = ptr;
    CvCapture *cap = arg->cap;
    image *buff = arg->buff;
    image *buff_letter =  arg->buff_letter;
    int buff_index = arg->buff_index;
    char *win_name = arg->window_name;
    IplImage *ipl = arg->ipl;

    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l,buff[0].w, buff[0].h, net->w, net->h, demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];

    sprintf(output_buf,"{\"objects\":[");
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes,
                    logger, output_buf);
    sprintf(output_buf+strlen(output_buf),"]}");

    reply = redisCommand(c,"LPUSH objectlist %s", output_buf);
    freeReplyObject(reply);
    memset(output_buf,0x0,sizeof(output_buf));

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;

    return 0;
}

void *fetch_in_thread(void *ptr)
{
    T_Thread_Arg *arg = ptr;
    CvCapture *cap = arg->cap;
    image *buff = arg->buff;
    image *buff_letter =  arg->buff_letter;
    int buff_index = arg->buff_index;
    int status = fill_image_from_stream_compress(cap, buff[buff_index], 0.5);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    T_Thread_Arg *arg = ptr;
    CvCapture *cap = arg->cap;
    image *buff = arg->buff;
    image *buff_letter =  arg->buff_letter;
    int buff_index = arg->buff_index;
    char *win_name = arg->window_name;
    IplImage *ipl = arg->ipl;

    show_image_cv(buff[(buff_index + 1)%3], win_name, ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}
void demo(type_param* param)
//void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    char *cfgfile = param->cfg;
    char *weightfile = param->weigths;
    float thresh = param->thresh;
    int cam_index = param->cam_index;
    const char *filename = param->filename;
    char **names = param->names;
    int classes = param->classes;
    int delay = param->frame_skip;
    char *prefix = param->prefix;
    int avg_frames = param->avg;
    float hier = param->hier_thresh;
    int w = param->width;
    int h = param->height;
    int frames = param->fps;
    int fullscreen = param->fullscreen;

    T_Thread_Arg thread_arg;

    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    memset(output_buf, 0x0, sizeof(output_buf));
    if (log4c_init()) {
        printf("log4c_init() failed");
        return;
    }
    logger = log4c_category_get("darknet");

    const char *hostname = "101.200.39.177";
    int port = 6379;

    struct timeval timeout = { 1, 500000 }; // 1.5 seconds
    c = redisConnectWithTimeout(hostname, port, timeout);

    if (c == NULL || c->err) {
        if (c) {
            printf("Connection error: %s\n", c->errstr);
            redisFree(c);
        } else {
            printf("Connection error: can't allocate redis context\n");
        }
        exit(1);
    }

    for(int i=0; i<NUM_VIDEO_CHANNEL;i++)
    {
        if(filename){
            printf("video file: %s\n", filename);
            cap[i] = cvCreateFileCapture(filename);
        }else{
            cap[0] = cvCaptureFromCAM(cam_index);

            if(w){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FRAME_WIDTH, w);
            }
            if(h){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FRAME_HEIGHT, h);
            }
            if(frames){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FPS, frames);
            }
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    for(int i=0; i<NUM_VIDEO_CHANNEL; i++) {
        buff[i][0] = get_image_from_stream_compress(cap[i], 0.5);
        buff[i][1] = get_image_from_stream_compress(cap[i], 0.5);
        buff[i][2] = get_image_from_stream_compress(cap[i], 0.5);
        buff_letter[i][0] = letterbox_image(buff[i][0], net->w, net->h);
        buff_letter[i][1] = letterbox_image(buff[i][1], net->w, net->h);
        buff_letter[i][2] = letterbox_image(buff[i][2], net->w, net->h);
        ipl[i] = cvCreateImage(cvSize(buff[i][0].w, buff[i][0].h), IPL_DEPTH_8U, buff[i][0].c);
    }
    int count = 0;

    if(!prefix){
        for(int i=0; i<NUM_VIDEO_CHANNEL; i++) {
            memset(window_name[i],0x0,sizeof(window_name[i]));
            sprintf(window_name[i],"%d",i);
            cvNamedWindow(window_name[i], CV_WINDOW_NORMAL);

            if (fullscreen) {
                cvSetWindowProperty(window_name[i], CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            } else {
                cvMoveWindow(window_name[i], 0, 0);
                cvResizeWindow(window_name[i], ipl[i]->width, ipl[i]->height);
            }
        }
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        for(int i=0; i<NUM_VIDEO_CHANNEL; i++)
        {
            buff_index[i] = (buff_index[i] + 1) %3;

            memcpy(thread_arg.buff, buff[i], sizeof(buff[i]));
            memcpy(thread_arg.buff_letter, buff_letter[i], sizeof(buff_letter[i]));
            thread_arg.cap = cap[i];
            thread_arg.buff_index = buff_index[i];
            thread_arg.ipl = ipl[i];
            thread_arg.window_name = window_name[i];

            if(pthread_create(&fetch_thread, 0, fetch_in_thread, &thread_arg)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, &thread_arg)) error("Thread creation failed");

            if(!prefix){
                fps = 1./(what_time_is_it_now() - demo_time);
                demo_time = what_time_is_it_now();
                display_in_thread(&thread_arg);
            }else{
                char name[256];
                sprintf(name, "%s_%08d", prefix, count);
                save_image(buff[i][(buff_index[i] + 1)%3], name);
            }
            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            ++count;
        }
    }

    if ( log4c_fini()){
        printf("log4c_fini() failed");
    }

    freeReplyObject(reply);

    /* Disconnects and frees the context */
    redisFree(c);

}

void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfg1, weight1, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    for(int i; i<NUM_VIDEO_CHANNEL; i++)
    {
        if(filename){
            printf("video file: %s\n", filename);
            cap[i] = cvCaptureFromFile(filename);
        }else{
            cap[i] = cvCaptureFromCAM(cam_index);

            if(w){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FRAME_WIDTH, w);
            }
            if(h){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FRAME_HEIGHT, h);
            }
            if(frames){
                cvSetCaptureProperty(cap[i], CV_CAP_PROP_FPS, frames);
            }
        }

        if(!cap[i]) error("Couldn't connect to webcam.\n");
    }

    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    for(int i=0; i<NUM_VIDEO_CHANNEL; i++) {

        buff[i][0] = get_image_from_stream_compress(cap[i], 1);
        buff[i][1] = get_image_from_stream_compress(cap[i], 1);
        buff[i][2] = get_image_from_stream_compress(cap[i], 1);
        buff_letter[i][0] = letterbox_image(buff[i][0], net->w, net->h);
        buff_letter[i][1] = letterbox_image(buff[i][0], net->w, net->h);
        buff_letter[i][2] = letterbox_image(buff[i][0], net->w, net->h);
        ipl[i] = cvCreateImage(cvSize(buff[i][0].w, buff[i][0].h), IPL_DEPTH_8U, buff[i][0].c);
    }

    int count = 0;
    char win_name[16];
    if(!prefix){
        for(int i=0; i<NUM_VIDEO_CHANNEL; i++) {
            sprintf(win_name,"%d",i);
            cvNamedWindow(win_name, CV_WINDOW_NORMAL);
            if (fullscreen) {
                cvSetWindowProperty(win_name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            } else {
                cvMoveWindow(win_name, 0, 0);
                cvResizeWindow(win_name, buff[i][0].w, buff[i][0].h);
            }
        }
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        for(int i=0; i<NUM_VIDEO_CHANNEL; i++) {
            buff_index[i] = (buff_index[i] + 1) % 3;
            if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
            if (!prefix) {
                fps = 1. / (what_time_is_it_now() - demo_time);
                demo_time = what_time_is_it_now();
                display_in_thread(0);
            } else {
                char name[256];
                sprintf(name, "%s_%08d", prefix, count);
                save_image(buff[i][(buff_index[i] + 1) % 3], name);
            }
            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);
            ++count;
        }
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

