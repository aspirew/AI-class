import glob
import logging
import sys
import time

import cv2
import os
import numpy as np
import uuid
import keyboard
# tensorflow deps

from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv1D, Conv2D, Dense, MaxPooling2D, MaxPool1D, Input, Flatten
from keras.metrics import Precision, Recall
import tensorflow as tf

import scipy.io.wavfile as wav
from speechpy.feature import mfcc
from speechpy.processing import cmvn

import pyaudio
import wave

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
INPUT_IMAGE = os.path.join('application_data', 'input_image')
VER_IMAGE = os.path.join('application_data', 'verification_images')

POS_PATH_SOUND = os.path.join('sound_data', 'positive') 
NEG_PATH_SOUND = os.path.join('sound_data', 'clips')
ANC_PATH_SOUND = os.path.join('sound_data', 'anchor')

def create_dirs():
    if (not os.path.exists(POS_PATH)):
        os.makedirs(POS_PATH)
    if (not os.path.exists(NEG_PATH)):
        os.makedirs(NEG_PATH)
    if (not os.path.exists(ANC_PATH)):
        os.makedirs(ANC_PATH)

    if (os.path.exists('lfw')):
        for dir in os.listdir('lfw'):
            for file in os.listdir(os.path.join(os.getcwd(), 'lfw', dir)):
                EX_PATH = os.path.join(os.getcwd(), 'lfw', dir, file)
                NEW_PATH = os.path.join(os.getcwd(), NEG_PATH, file)
                os.replace(EX_PATH, NEW_PATH)

def data_aug(img):
    data = []
    for _ in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

# webcam connection
# different camera ID might be necessary
def run_camera_for_collection():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()

        # cut image down to 250x250
        frame = frame[350:350 + 250, 550:550 + 250, :]
        cv2.imshow('Image Collection', frame)

        # save anchor image with a press
        if cv2.waitKey(1) and keyboard.is_pressed('a'):
            imgname = os.path.join(os.getcwd(), ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # save positive image with p press
        if cv2.waitKey(1) and keyboard.is_pressed('p'):
            imgname = os.path.join(os.getcwd(), POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # quit with q press
        if cv2.waitKey(1) and keyboard.is_pressed('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def add_augmented_images():
    for file_name in os.listdir(os.path.join(os.getcwd(), POS_PATH)):
        img_path = os.path.join(os.getcwd(), POS_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img) 

        for image in augmented_images:
            cv2.imwrite(os.path.join(os.getcwd(), POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

# function to create numpy equivalent of image
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    # resize value to 100x100 (as given in paper)
    img = tf.image.resize(img, (100, 100))
    # scale pixel value to be between 0 and 1
    img = img / 255.0
    return img


def preprocess_and_label_all_images():
    anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(3000)

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


def make_embedding_for_image():
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding, shape):
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=shape)
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=shape)
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


def training(train_data, embedding, shape):
    siamese_model = make_siamese_model(embedding, shape)
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
        for epoch in range(1, EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))
            
            # Creating a metric object 
            r = Recall()
            p = Precision()
            
            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat) 
                progbar.update(idx+1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())
            
            # Save checkpoints
            if epoch % 10 == 0: 
                checkpoint.save(file_prefix=checkpoint_prefix)

    EPOCHS = 50

    train(train_data, EPOCHS)
    return siamese_model


def evaluate(model, test_data):
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true,yhat) 

    print(r.result().numpy(), p.result().numpy())


def save_model(model, name):
    model.save(name)


def reload_model(filename):
    return tf.keras.models.load_model(filename, custom_objects={'L1Dist': L1Dist,
                                                                'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


def verify(model, detection_threshold, verification_threshold, login):
    # Build results array
    results = []
    for image in os.listdir(os.path.join(os.getcwd(), 'application_data', login, 'verification_images')):
        input_img = preprocess(os.path.join(os.getcwd(), 'application_data', login, 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join(os.getcwd(), 'application_data', login, 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose = 0)
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join(os.getcwd(), 'application_data', login, 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


def real_time_verification(model, login):
    cap = cv2.VideoCapture(0)
    print("Ustaw się tak, żeby twoja twarz była widoczna w oknie kamerki widocznym na ekranie")
    print("Kiedy będziesz gotowy, wciśnij przycisk v, w celu weryfikacji...")
    chances = 0
    verified = False
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[350:350 + 250, 550:550 + 250, :]

        cv2.imshow('Verification', frame)
        cv2.setWindowProperty('Verification', cv2.WND_PROP_TOPMOST, 1)

        # trigger verification on 'v' press
        if cv2.waitKey(10) and keyboard.is_pressed('v'):
            print("Trwa weryfikacja...")
            cv2.imwrite(os.path.join(os.getcwd(), 'application_data', login, 'input_image', 'input_image.jpg'), frame)
            results, verified = verify(model, 0.5, 0.5, login)
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

def preprocess_sound(sound):
    rate, sig = wav.read(sound)
    mfcc_feat = mfcc(sig,rate, fft_length=2048)
    norm_features = cmvn(mfcc_feat)
    return tf.convert_to_tensor(norm_features[20:290])

def preprocess_and_label_all_sounds():
    anchor = [f for f in glob.glob(ANC_PATH_SOUND + "/*.wav")]
    positive = [f for f in glob.glob(POS_PATH_SOUND + "/*.wav")]
    negative = [f for f in glob.glob(NEG_PATH_SOUND + "/*.wav")]
    negative = negative[:50]
        
    def preprocess_all_sounds(list_of_sounds):
        values = []
        for sound in list_of_sounds:
            values.append(preprocess_sound(sound))
        return values
    
    processed_anchor = tf.data.Dataset.from_tensor_slices(preprocess_all_sounds(anchor))
    processed_positive = tf.data.Dataset.from_tensor_slices(preprocess_all_sounds(positive))
    processed_negative = tf.data.Dataset.from_tensor_slices(preprocess_all_sounds(negative))
        
    positives = tf.data.Dataset.zip((processed_anchor, processed_positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((processed_anchor, processed_negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)
    
    # preprocess all images and shuffle negatives with positives
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


def make_embedding_for_recording():
    inp = Input(shape=(270,13), name='input_sample')
    
    # First block
    c1 = Conv1D(64, 10, activation='relu')(inp)
    m1 = MaxPool1D(64, 2, padding='same')(c1)
    
    # Second block
    c2 = Conv1D(128, 7, activation='relu')(m1)
    m2 = MaxPool1D(64, 2, padding='same')(c2)
    
    # Third block 
    c3 = Conv1D(128, 4, activation='relu')(m2)
    m3 = MaxPool1D(64, 2, padding='same')(c3)
    
    # Final embedding block
    c4 = Conv1D(256, 4, activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

def verify_sound(model, detection_threshold, verification_threshold, login):
    # Build results array
    results = []
    for sound in os.listdir(os.path.join(os.getcwd(), 'application_data', login, 'verification_sounds')):
        input_sound = preprocess_sound(os.path.join(os.getcwd(), 'application_data', login, 'input_sound', 'input_sound.wav'))
        validation_sound = preprocess_sound(os.path.join(os.getcwd(), 'application_data', login, 'verification_sounds', sound))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_sound, validation_sound], axis=1)), verbose = 0)
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join(os.getcwd(), 'application_data', login, 'verification_sounds'))) 
    verified = verification > verification_threshold
    
    return results, verified

def real_time_sound_verification(model, login):
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    
    print("Kiedy będziesz gotowy, mów do mikrofonu głośno i wyraźnie przez około 3 sekundy")
    print("Wciśnij jednokrotnie przycisk v, a aplikacja zacznie rejestrować dźwięk...")
    chances = 0
    verified = False
    while chances < 3:
        # trigger recording on 'v' press
        if keyboard.is_pressed('v'):
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
            print("Trwa nagrywanie...")
            frames = []
            
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            waveFile = wave.open(os.path.join(os.getcwd(), 'application_data', login, 'input_sound', 'input_sound.wav'), 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            print("Nagrywanie zakończone. Trwa weryfikacja...")
            
            results, verified = verify_sound(model, 0.5, 0.5, login)
            if verified:
                print("Udało się poprawnie zweryfikować głos użytkownika!")
                return True
            else:
                chances += 1
                print(f'Nie udało się poprawnie zweryfikować danego użytkownika, wykorzystane szanse {chances}/3')
                print("Kiedy będziesz gotowy, mów do mikrofonu głośno i wyraźnie przez około 3 sekundy")
                print("Wciśnij jednokrotnie przycisk v, a aplikacja zacznie rejestrować dźwięk...")
            if chances == 3:
                print("Przekroczono dostępną liczbę prób")
                return False
    return verified

def main_program():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    print("Cześć!")
    login = ""
    while not os.path.exists(os.path.join(os.getcwd(), 'models', login, 'image.h5')):
        login = input("Wprowadź swój login: ")
        if not os.path.exists(os.path.join(os.getcwd(),'models', login, 'image.h5')):
            print("Podano niepoprawny login")
    modelImage = reload_model(os.path.join(os.getcwd(), 'models', login, 'image.h5'))
    modelSound = reload_model(os.path.join(os.getcwd(), 'models', login, 'sound.h5'))
    print("Za 3 sekundy nastąpi uruchomienie kamery")
    for i in reversed(range(3)):
        print(i + 1)
        time.sleep(1)
    if real_time_verification(modelImage, login):
        if real_time_sound_verification(modelSound, login):
            print(f"Witaj {login}!")
            print("Nastąpi zamknięcie aplikacji...")
            return
    print("Niepowodzenie logowania!")
    print("Nastąpi zamknięcie aplikacji...")

main_program()
# create_dirs()
# run_camera_for_collection()
# add_augmented_images()
# train_data, test_data = preprocess_and_label_all_images()
# model = training(train_data, make_embedding_for_image(), (100, 100, 3))

# evaluate(model, test_data)
# save_model(model, 'rafal2.h5')
# model = reload_model('mati.h5')
# real_time_verification(model)

# train_data, test_data = preprocess_and_label_all_sounds()
# model = training(train_data, make_embedding_for_recording(), (270,13))
# evaluate(model, test_data)
# save_model(model, 'rafal_sound2.h5')
# model = reload_model('rafal_sound2.h5')
# print(verify_sound(model, 0.5, 0.5))
# train_data = preprocess_and_label_sounds(os.path.join(NEG_PATH_SOUND, 'common_voice_pl_34938116.mp3'))
# training(train_data, make_embedding_for_recording(), (292, 13))
