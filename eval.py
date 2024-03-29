import cv2
import time
import math
import os
import argparse
import numpy as np
import tensorflow as tf
from keras.models import load_model, model_from_json

import nms
from data_processor import restore_rectangle

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='')
parser.add_argument('--model_path', type=str, default='/home/list_99/Python/EASTmodel/model-120.h5')
parser.add_argument('--image',type=str,default='/home/list_99/Python/opencv-text-detection/dataset/testing_data/images/82200067_0069.png')
FLAGS = parser.parse_args()




def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms.nms_locality(boxes.astype(np.float64), nms_thres)
    print(boxes.shape)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):

    # load trained model
    json_file = open(os.path.join('/'.join(FLAGS.model_path.split('/')[0:-1]), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': 2})
    model.load_weights(FLAGS.model_path)

    img_file = FLAGS.image

    img = cv2.imread(img_file)[:, :, ::-1]  #BGR -> RGB
    start_time = time.time()
    img_resized, (ratio_h, ratio_w) = resize_image(img)

    img_resized = (img_resized / 127.5) - 1

    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()

    # feed image into model
    score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])

    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score_map, geo_map=geo_map, timer=timer)
    print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        img_file, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
    print(boxes.shape)
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    duration = time.time() - start_time
    print('[timing] {}'.format(duration))


    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            #if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                #continue
            cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0))

        
    cv2.imshow('test', img[:, :, ::-1])
    cv2.imwrite('test2.png',img[:, :, ::-1])
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
