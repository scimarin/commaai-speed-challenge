#!/usr/bin/python

import numpy as np
import cv2 as cv


def cut_frame(frame):
    left_cutout = 5
    right_cutout = 35

    bottom_cutout = 130
    top_cutout = 110

    height = frame.shape[0]
    width = frame.shape[1]

    frame = frame[top_cutout:height - bottom_cutout, left_cutout:width - right_cutout,:]

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    return frame


def apply_filter(frame):
    frame = cv.GaussianBlur(frame, (3, 3), 0)

    return frame


def change_brightness(frame, factor):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    hsv[...,2] = hsv[...,2] * factor
    frame = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    return frame


def compute_dense_optical_flow(prev_frame, cur_frame):
    gray_prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    gray_cur_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(cur_frame)

    hsv[...,1] = 255 # fully saturated

    flow = cv.calcOpticalFlowFarneback(gray_prev_frame,
                                       gray_cur_frame,
                                       None, 0.5, 3, 15, 3, 5, 1.2,
                                       cv.OPTFLOW_FARNEBACK_GAUSSIAN)

    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    hsv[...,2] = hsv[...,2]

    rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    return rgb

