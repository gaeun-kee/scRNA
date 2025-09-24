import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten,
    Dense, Dropout, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# -----------------------------
# 1D CNN 모델
# -----------------------------
def build_cnn1d(input_shape=(12413, 1), dropout_ratio=0.3, learning_rate=1e-4):
    """
    1D CNN 모델 생성

    Args:
        input_shape (tuple): 입력 데이터 형태 (features, 1)
        dropout_ratio (float): Dropout 비율
        learning_rate (float): Adam optimizer 학습률

    Returns:
        model (keras.Model): 컴파일된 CNN1D 모델
    """
    inputs = keras.Input(shape=input_shape)

    x = Conv1D(128, 15, padding="valid")(inputs)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, 15, padding="valid")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(32, 5, padding="valid")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(dropout_ratio)(x)

    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)

    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    return model


# -----------------------------
# 2D CNN 모델
# -----------------------------
def build_cnn2d(input_shape=(383, 186, 1), dropout_ratio=0.3, learning_rate=1e-4):
    """
    2D CNN 모델 생성

    Args:
        input_shape (tuple): 입력 데이터 형태 (rows, cols, 1)
        dropout_ratio (float): Dropout 비율
        learning_rate (float): Adam optimizer 학습률

    Returns:
        model (keras.Model): 컴파일된 CNN2D 모델
    """
    inputs = keras.Input(shape=input_shape)

    x = Conv2D(128, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_ratio)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dropout(dropout_ratio)(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_ratio)(x)

    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    return model