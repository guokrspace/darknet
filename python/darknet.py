from ctypes import *
import math
import random
import numpy as np
import cv2
from PIL import Image

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/kai/darknet/python/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

def detect_frame(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    net = load_net(c_char_p(b'../cfg/tiny-yolo.cfg'), c_char_p(b'../tiny-yolo.weights'), c_int(0))
    meta = load_meta(c_char_p(b'../cfg/coco.data'))
    # r = detect(net, meta, c_char_p(b'../data/dog.jpg'))
    # im = load_image(c_char_p(b'../data/dog.jpg'),0,0)

    # frame = cv2.imread('../data/dog.jpg')
    #
    # w = frame.shape[1];
    # h = frame.shape[0];
    # c = frame.shape[2];
    #
    # # frame1 = np.zeros(1327104)
    # #
    # # for i in range(1327104):
    # #     frame1[i] = im.data[i] * 256
    #
    # # frame1 = frame1.reshape(-1,576*768)
    # # frame1 = np.flip(frame1,0)
    # # frame1 = frame1.T
    # # frame1 = frame1.reshape(576,768,3)
    #
    # frame1 = frame.reshape(576*768,-1)
    # frame1 = frame1.T
    # frame1 = np.flip(frame1, 0)
    # frame1 = frame1.reshape(-1)
    # frame1 = np.divide(frame1,255)
    #
    # # for i in range(3):
    # #     for j in range (768):
    # #         for k in range (576):
    # #             frame
    # #
    # c_float_p = POINTER(c_float)
    # data = np.array(frame1)
    # data = data.astype(np.float32)
    # data_p = data.ctypes.data_as(c_float_p)
    # im1 = IMAGE(w,h,c, data_p)
    # r1 = detect_frame(net, meta, im1)
    #
    #
    # # for i in range(1000):
    #     # print(im.data[i])
    #     # print(frame1[i])
    #     # print('')
    # # Display the resulting frame
    # for box in r1:
    #     x = box[2][0]
    #     y = box[2][1]
    #     w = box[2][2]
    #     h = box[2][3]
    #     name = box[0].decode()
    #
    #     cv2.rectangle(frame,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,0),3)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(frame, name, (int(x-w/2+10),int(y-h/2+10)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #
    # cv2.imshow('dst_rt', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    cap = cv2.VideoCapture("rtsp://admin:admin@10.168.5.155:554/cam/realmonitor?channel=1&subtype=1")
    #cap = cv2.VideoCapture('road.mp4')

    while (True):
        ret, frame = cap.read();

        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

        # if frame == None:
        #     continue

        w = frame.shape[1];
        h = frame.shape[0];
        c = frame.shape[2];

        # im = lib.get_image_from_stream(cap)

        frame_hat = frame.reshape(w * h, -1)
        frame_hat = frame_hat.T
        frame_hat = np.flip(frame_hat, 0)
        frame_hat = frame_hat.reshape(-1)
        frame_hat = np.divide(frame_hat, 255)
        # frame = np.array(frame)

        # NOTE: w x h x c is intepreted to w x( h x c) in one dim array
        #       Reshape to c x h x w to make is c ( h x w)
        # frame = frame.reshape(w, h, c);
        #
        # frame = np.divide(frame,255.)

        c_float_p = POINTER(c_float)
        data = np.array(frame_hat)
        data = data.astype(np.float32)
        data_p = data.ctypes.data_as(c_float_p)

        im = IMAGE(w, h, c, data_p)

        r = detect_frame(net, meta, im)


    #     # Display the resulting frame
        for box in r:
            print(r)
            x = box[2][0]
            y = box[2][1]
            w = box[2][2]
            h = box[2][3]
            name = box[0].decode()

            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (int(x - w / 2 + 10), int(y - h / 2 + 10)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # frame = frame.reshape(w, h, c);
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
