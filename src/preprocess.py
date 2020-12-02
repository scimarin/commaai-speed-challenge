#!/usr/bin/python

import numpy as np
import cv2 as cv


no_samples = 20400
framepath = '../data/test_pics/frame_{}.png'
# framepath = '../data/train_pics/frame_{}.png'


if __name__ == '__main__':
    cap = cv.VideoCapture('./data/test.mp4')
    # cap = cv.VideoCapture('./data/train.mp4')

    valid, frame = cap.read()
    cv.imwrite(framepath.format(0), frame)

    i = 1
    while valid:
        valid, frame = cap.read()

        if not valid or frame is None:
            break

        cv.imwrite(framepath.format(i), frame)
        i += 1
        print('processed frame {}'.format(i))
