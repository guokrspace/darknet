#include "darknet.h"
#include "image.h"
//
// Created by kai on 18-1-24.
//

network * init_network(char *cfgfile, char*weightfile)
{
    printf("************Start init_network*********************\n");
    printf("%s\n",cfgfile);
    printf("%s\n",weightfile);
    network *net;
    net = load_network(cfgfile, weightfile, 0);

    set_batch_network(net, 1);

    printf("************End init_network*********************\n");
    return net;
}

void check_network(network *net)
{
    printf("network.w: %d",net->w);
    printf("network.h: %d",net->h);
    printf("network.c: %d\n",net->c);
}

image decode_frame(void *input_buf, int w, int h, int c, network *net)
{

    printf("Trace decode_frame start\n");
    check_network(net);
    IplImage *dst = cvCreateImage(cvSize(w, h), 8 ,c);
    image img = make_image(w, h, c);
    unsigned char *src = input_buf;

    printf("Trace decode_frame 0\n");
    printf("%x",src);
    printf("Trace decode_frame 000\n");

    for(int i=0; i<1; i++)
        printf("%d\n",src[i]);
    printf("Trace decode_frame 00\n");
    int i,j,k;
    for(i=0; i<c; i++)
        for(j=0;j<h;j++)
            for(k=0;k<w;k++)
            {
                float p = src[i * w * h + j * w + k];
                img.data[i * w * h + j * w + k] = p;
            }

    printf("Trace decode_frame 1\n");

    rgbgr_image(img);

    image buff = img;
    image buff_letter = letterbox_image(img, net->w, net->h);

    float nms = .4;

    printf("Trace decode_frame 2\n");

    layer l = net->layers[net->n-1];
    float *X = buff_letter.data;
    float *prediction = network_predict(net, X);
    box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    printf("%d, %d, %d, %f",l.w,l.n,l.h,*l.output);

    l.output = prediction;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, 0, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l,buff.w, buff.h, net->w, net->h, 0, probs, boxes, 0, 0, 0, 0.5, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);



    list *options = read_data_cfg("/home/kai/darknet/cfg/coco.data");
    int classes = option_find_int(options, "classes", 20);
    char *name_list = option_find_str(options, "names", "/home/kai/darknet/data/names.list");
    char **names = get_labels(name_list);
    image **alphabet = load_alphabet();

    int detections = l.n*l.w*l.h;

    image display = buff;

    draw_detections_local(display, detections, 0, boxes, probs, 0, names, alphabet, classes);

    printf("Trace decode_frame end");

    return display;

}

void draw_detections_local(image im, int num, float thresh, box *boxes, float **probs, float **masks, char **names,
                     image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
            if (probs[i][j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
                break; //???? Added by YK
            }
        }
        if(class >= 0){
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;


            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, labelstr, (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (masks){
                image mask = float_to_image(14, 14, 1, masks[i]);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }


}


