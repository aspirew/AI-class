import logging
import sys
import time

import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid
import keyboard

# tensorflow deps
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
INPUT_IMAGE = os.path.join('application_data', 'input_image')
VER_IMAGE = os.path.join('application_data', 'verification_images')


def create_dirs():
    if (not os.path.exists(POS_PATH)):
        os.makedirs(POS_PATH)
    if (not os.path.exists(NEG_PATH)):
        os.makedirs(NEG_PATH)
    if (not os.path.exists(ANC_PATH)):
        os.makedirs(ANC_PATH)
    if (not os.path.exists(INPUT_IMAGE)):
        os.makedirs(INPUT_IMAGE)
    if (not os.path.exists(VER_IMAGE)):
        os.makedirs(VER_IMAGE)

    if (os.path.exists('lfw')):
        for dir in os.listdir('lfw'):
            for file in os.listdir(os.path.join('lfw', dir)):
                EX_PATH = os.path.join('lfw', dir, file)
                NEW_PATH = os.path.join(NEG_PATH, file)
                os.replace(EX_PATH, NEW_PATH)


# webcam connection
# different camera ID might be necessary
def run_camera_for_collection():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        XD, frame = cap.read()

        # cut image down to 250x250
        frame = frame[0:250, 0:250, :]
        cv2.imshow('Image Collection', frame)

        # save anchor image with a press
        if cv2.waitKey(1) and keyboard.is_pressed('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # save positive image with p press
        if cv2.waitKey(1) and keyboard.is_pressed('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # quit with q press
        if cv2.waitKey(1) and keyboard.is_pressed('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# function to create numpy equivalent of image
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img, 3)
    # resize value to 100x100 (as given in paper)
    img = tf.image.resize(img, (100, 100))
    # scale pixel value to be between 0 and 1
    img = img / 255.0
    return img


def preprocess_and_label_all_images():
    anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(100)
    positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(100)
    negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(100)

    dir_test = anchor.as_numpy_iterator()
    dir_test.next()

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    def preprocess_twin(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)

    # preprocess all images and shuffle negatives with positives
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    train_data = data.take(round(len(data) * .7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data) * .7))
    test_data = test_data.take(round(len(data) * .3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data, test_data


def make_embedding():
    input = Input(shape=(100, 100, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(input)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[input], outputs=[d1], name='embedding')


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model():
    embedding = make_embedding()
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


def training(train_data):
    siamese_model = make_siamese_model()
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = './training_checkpoints'
    if (not os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    @tf.function
    def train_step(batch):

        with tf.GradientTape() as tape:
            # get anchor and positive/negative image
            x = batch[:2]
            # get label
            y = batch[2]

            # forward pass
            yhat = siamese_model(x, training=True)

            # calculate loss
            loss = binary_cross_loss(y, yhat)

        # calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)

        # calculate updated weights
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        return loss

    def train(data, EPOCHS):
        for epoch in range(1, EPOCHS):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))

            for idx, batch in enumerate(data):
                train_step(batch)
                progbar.update(idx + 1)

            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

    EPOCHS = 3

    train(train_data, EPOCHS)
    return siamese_model


def evaluate(model, test_data):
    from keras.metrics import Precision, Recall
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    y_hat = model.predict([test_input, test_val])
    m = Recall()
    m.update_state(y_true, y_hat)
    print(m.result().numpy())


def save_model(model, name):
    model.save(name)


def reload_model(filename):
    return tf.keras.models.load_model(filename, custom_objects={'L1Dist': L1Dist,
                                                                'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


def verify(model, detection_threshold, verification_treshold):
    results = []
    for image in os.listdir(VER_IMAGE):
        input_img = preprocess(os.path.join(INPUT_IMAGE, 'input_image.jpg'))
        validation_img = preprocess(os.path.join(VER_IMAGE, image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
        # print(result)
        results.append(result)

    # detection threshold: metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    verification = detection / len(os.listdir(VER_IMAGE))

    # verification threshold: proportion of positive predicitons / total positive samples
    verified = verification > verification_treshold

    return results, verified


def real_time_verification(model):
    cap = cv2.VideoCapture(0)
    print("Ustaw się tak, żeby twoja twarz była widoczna w oknie kamerki widocznym na ekranie")
    print("Kiedy będziesz gotowy, wciśnij przycisk v, w celu weryfikacji...")
    chances = 0
    verified = False
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[0:250, 0:250, :]

        cv2.imshow('Verification', frame)
        cv2.setWindowProperty('Verification', cv2.WND_PROP_TOPMOST, 1)

        # trigger verification on 'v' press
        if cv2.waitKey(10) and keyboard.is_pressed('v'):
            print("Trwa weryfikacja...")
            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
            results, verified = verify(model, 0.5, 0.5)
            # print(verified)
            if verified:
                print("Udało się poprawnie zweryfikować twarz użytkownika")
                return True
            else:
                chances += 1
                print(f'Nie udało się poprawnie zweryfikować danego użytkownika, wykorzystane szanse {chances}/3')
            if chances == 3:
                print("Przekroczono dostępną liczbę prób")
                return False
        if cv2.waitKey(10) and keyboard.is_pressed('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return verified


def main_program():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    print("Cześć!")
    login = ""
    while not os.path.exists(f'{login}.h5'):
        login = input("Wprowadź swój login: ")
        if not os.path.exists(f'{login}.h5'):
            print("Podano niepoprawny login")
    model = reload_model(f'{login}.h5')
    print("Za 3 sekundy nastąpi uruchomienie kamery")
    for i in reversed(range(3)):
        print(i + 1)
        time.sleep(1)
    real_time_verification(model)


main_program()
# create_dirs()
# run_camera_for_collection()
# train_data, test_data = preprocess_and_label_all_images()
# model = training(train_data)
# model = reload_model('mati.h5')

# evaluate(model, test_data)
# save_model(model, 'mati.h5')
# model = reload_model('mati.h5')
# real_time_verification(model)
