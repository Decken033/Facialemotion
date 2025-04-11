import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from sklearn import model_selection
from math import ceil
from datetime import datetime

from tensorflow.keras.regularizers import l2  # 添加L2正则化
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 添加回调函数

# Loads csv files and appends pixels to X and labels to y
def preprocess_data():
    # data = pd.read_csv('fer2013.csv')

    # 读取train, test, val文件
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('val.csv')
    test_data = pd.read_csv('test.csv')
    data = pd.concat([train_data, val_data, test_data], ignore_index=True)


    labels = pd.read_csv('fer2013new.csv')

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y, len(train_data), len(val_data), len(test_data)


def clean_data_and_normalize(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        #增加缩放，亮度变换，剪切
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        #brightness_range=[0.2, 1.5],
    )
    datagen.fit(x_train)
    return datagen


def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
        # plt.xlabel(class_names[y_train[i]])
    plt.show()



def define_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64
    # 添加L2正则化参数
    l2_reg = 0.0001

    model = Sequential()

    # 1st stage - 保持原来的valid padding
    model.add(Conv2D(num_features, kernel_size=(3, 3), padding='valid', input_shape=input_shape,
                     kernel_regularizer=l2(l2_reg)))  # 保持原来的valid padding
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(num_features, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(l2_reg)))  # 保持原来的valid padding
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # 2nd stage - 保持原来的valid padding
    model.add(Conv2D(num_features, (3, 3), padding='valid', kernel_regularizer=l2(l2_reg)))  # 保持原来的valid padding
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(num_features, (3, 3), padding='valid', kernel_regularizer=l2(l2_reg)))  # 保持原来的valid padding
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # 3rd stage - 这里开始使用same padding防止尺寸问题
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # 4th stage - 继续使用same padding
    model.add(Conv2D(2 * num_features, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(2 * num_features, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # 5th stage - 继续使用same padding
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(l2_reg)))  # 改为same
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    return model

def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    plt.show()




def save_model_and_weights(model, test_acc, save_dir='Saved-Models', model_name='cnn'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'{model_name}_{timestamp}_acc_{test_acc:.4f}'
    json_path = os.path.join(save_dir, f'{base_name}_structure.json')
    h5_path = os.path.join(save_dir, f'{base_name}_weights.h5')

    model_json = model.to_json()
    with open(json_path, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(h5_path)

    print(f'Model saved to {json_path} and weights saved to {h5_path}')


def load_model_and_weights(model_path, weights_path):
    # Loading JSON model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Loading weights
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('Model and weights are loaded and compiled.')


def run_model():
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    X, y, train_size, val_size, test_size = preprocess_data()
    X, y = clean_data_and_normalize(X, y)
    # x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)

    # 直接使用数据集
    x_train, y_train = X[:train_size], y[:train_size]
    x_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    x_test, y_test = X[train_size + val_size:], y[train_size + val_size:]


    datagen = data_augmentation(x_train)

    epochs = 100
    batch_size = 32

    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    print("X_val shape: " + str(x_val.shape))
    print("Y_val shape: " + str(y_val.shape))

    # 添加回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )


    # Training model from scratch
    model = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=len(x_train) // batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, reduce_lr],# 使用回调函数
                        verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    plot_acc_loss(history)
    save_model_and_weights(model, test_acc)


run_model()
