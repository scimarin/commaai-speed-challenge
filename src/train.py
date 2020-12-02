#!/usr/bin/python

import numpy as np
import cv2 as cv
from tensorflow import keras

import process_frame as pf
import model


no_samples = 20400
framepath = '../data/train_pics/frame_{}.png'

seeds = [1, 2, 3, 4, 5, 12, 15]


def generate_training_data(data, batch_size):
    while True:
        frame_batch = []
        speed_batch = []

        for batch_ix in range(batch_size):
            ix = np.random.randint(1, len(data) - 1)

            prev_ix = int(data[ix - 1][0][0])
            prev_frame = cv.imread(framepath.format(prev_ix))
            prev_speed = float(data[ix - 1][0][1])

            cur_ix = int(data[ix][0][0])
            cur_frame = cv.imread(framepath.format(cur_ix))
            cur_speed = float(data[ix][0][1])

            next_ix = int(data[ix + 1][0][0])
            next_frame = cv.imread(framepath.format(next_ix))
            next_speed = float(data[ix + 1][0][1])

            if cur_ix - prev_ix == 1:
                frame1, frame1_speed = prev_frame, prev_speed
                frame2, frame2_speed = cur_frame, cur_speed
            elif next_ix - cur_ix == 1:
                frame1, frame1_speed = cur_frame, cur_speed
                frame2, frame2_speed = next_frame, next_speed
            else:
                raise Exception('Incorrect frame ordering')

            # augment the frames
            frame1, frame2 = pf.cut_frame(frame1), pf.cut_frame(frame2)
            frame1, frame2 = pf.apply_filter(frame1), pf.apply_filter(frame2)

            factor = 0.3 + np.random.uniform()
            frame1, frame2 = pf.change_brightness(frame1, factor), pf.change_brightness(frame2, factor)

            # compute dense optical flow
            flow_frame = pf.compute_dense_optical_flow(frame1, frame2)
            flow_speed = np.mean([frame1_speed, frame2_speed])

            # normalize and resize
            flow_frame = cv.normalize(flow_frame, None, alpha=-1, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            flow_frame = cv.resize(flow_frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

            frame_batch.append(flow_frame)
            speed_batch.append(flow_speed)

        yield np.array(frame_batch), np.array(speed_batch)


def generate_validation_data(data):
    while True:
        for ix in range(1, len(data)):
            frame1_ix = int(data[ix - 1][0][0])
            frame1 = cv.imread(framepath.format(frame1_ix))
            frame1_speed = float(data[ix - 1][0][1])

            frame2_ix = int(data[ix][0][0])
            frame2 = cv.imread(framepath.format(frame2_ix))
            frame2_speed = float(data[ix][0][1])

            # augment the frames
            frame1, frame2 = pf.cut_frame(frame1), pf.cut_frame(frame2)
            frame1, frame2 = pf.apply_filter(frame1), pf.apply_filter(frame2)

            # compute dense optical flow
            flow_frame = pf.compute_dense_optical_flow(frame1, frame2)
            flow_speed = np.mean([frame1_speed, frame2_speed])

            # normalize and resize
            flow_frame = cv.normalize(flow_frame, None, alpha=-1, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            flow_frame = cv.resize(flow_frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

            yield np.array([flow_frame]), np.array([[flow_speed]])


def prepare_data():
    # prepare training and validation data: 20% validation, 80% training, distributed randomly
    train_data, valid_data = [], []

    with open('../data/train.txt', 'r') as fin:
        speed = np.array([float(val.strip()) for val in fin.readlines()])

    np.random.seed(seeds[4])

    samples = np.random.permutation(np.arange(no_samples - 1))

    for ix in samples:
        prob = np.random.randint(0, 11)

        if prob < 2:
            valid_data.append([[ix, speed[ix]]])
            valid_data.append([[ix + 1, speed[ix + 1]]])
        else:
            train_data.append([[ix, speed[ix]]])
            train_data.append([[ix + 1, speed[ix + 1]]])

    return np.array(train_data), np.array(valid_data)


def train(model, train_generator, valid_generator, epochs, steps_per_epoch):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.2,
        patience=2,
        mode='min',
        verbose=1,
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='min',
        save_weights_only=True,
    )

    valid_steps = len(valid_data)

    print('training model...')

    history = model.fit(
        x=train_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        validation_data=valid_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=valid_steps,
    )

    return history


if __name__ == '__main__':
    train_data, valid_data = prepare_data()

    train_generator = generate_training_data(train_data, batch_size=32)
    valid_generator = generate_validation_data(valid_data)

    model = model.create(learning_rate=1e-4)
    train(model, train_generator, valid_generator, 50, 500)

