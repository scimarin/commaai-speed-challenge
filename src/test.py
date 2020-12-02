#!/usr/bin/python

import numpy as np
import cv2 as cv
from tensorflow import keras

import model
import process_frame as pf


no_samples = 10798
framepath = '../data/test_pics/frame_{}.png'


def generate_test_data():
    for ix in range(no_samples - 1):
        frame1 = cv.imread(framepath.format(ix))
        frame2 = cv.imread(framepath.format(ix + 1))

        # augment the frames
        frame1, frame2 = pf.cut_frame(frame1), pf.cut_frame(frame2)
        frame1, frame2 = pf.apply_filter(frame1), pf.apply_filter(frame2)

        # compute dense optical flow
        flow_frame = pf.compute_dense_optical_flow(frame1, frame2)

        # normalize and resize
        flow_frame = cv.normalize(flow_frame, None, alpha=-1, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        flow_frame = cv.resize(flow_frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

        print('computed frame {}'.format(ix))
        yield np.array([flow_frame])


if __name__ == '__main__':
    test_data = generate_test_data()

    model = model.create()
    pred = model.predict(test_data, steps=no_samples - 1)

    with open('preds.txt', 'w') as fout:
        for p in pred:
            fout.write(p)
    print(pred)
