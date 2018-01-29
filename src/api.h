//
// Created by kai on 18-1-24.
//

#include "network.h"

#ifndef DARKNET_API_H
network * init_network(char *cfgfile, char*weightfile);
image decode_frame(network *net, void *input_buf, int w, int h, int c);

#define DARKNET_API_H

#endif //DARKNET_API_H
