import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def set_seed(seed=42):
    """
    재현성을 위한 시드 고정
    """
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    save_path="./models/best_model.h5",
    epochs=30,
    batch_size=64,
    patience=3,
    monitor="val_loss",
    mode="min",
    return_best=True
):
    """
    공통 학습 루프 (1D/2D CNN 공용)

    Args:
        model (keras.Model): 학습할 Keras 모델
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        save_path (str): 체크포인트 저장 경로
        epochs (int): 학습 epoch 수
        batch_size (int): 배치 크기
        patience (int): EarlyStopping patience
        monitor (str): 모니터링 지표
        mode (str): "min" 또는 "max"
        return_best (bool): True면 best checkpoint 모델 로드 후 반환

    Returns:
        model, history
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    es = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=1)
    mc = ModelCheckpoint(save_path, monitor=monitor, mode=mode, save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc]
    )

    if return_best and os.path.exists(save_path):
        model.load_weights(save_path)
        print(f"Best model loaded from {save_path}")

    return model, history
