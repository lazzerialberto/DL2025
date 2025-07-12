import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121, Xception 

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

print("--- GPU Availability Check (TensorFlow 2.x) ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow can access {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
else:
    print("TensorFlow cannot access a GPU. Running on CPU.")

print("Loading dataset... ")
# --- CONFIG --- WITH CLASS WEIGHTING ---
IMG_WIDTH = 320
IMG_HEIGHT = 320
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

base_dir = 'data_redistributed_stratified'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# --- Dataset di base ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',  # 0 e 1
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False
)

def grayscale_to_rgb_gen(gen):
    for x_batch, y_batch in gen:
        x_rgb = np.repeat(x_batch, 3, axis=-1)
        yield x_rgb, y_batch
        
def preprocess(img, label):
    img = tf.image.grayscale_to_rgb(img)        # from (H,W,1) to (H,W,3)
    img = tf.cast(img, tf.float32) / 255.0       # normalization
    return img, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(0.05),
    tf.keras.layers.RandomFlip("horizontal"),
])

def augment(img, label):
    img = data_augmentation(img)
    return img, label

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

y_train = np.concatenate([y.numpy().flatten() for _, y in train_ds])

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)
print("Loading dataset completed.")

print("--- Model Definition ---")
def build_and_train(hparams):
    tf.keras.backend.clear_session()
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(320, 320, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(320, 320, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(hparams['dense_units'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hparams['dropout'])(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=hparams['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        epochs=5,
        verbose=0
    )

    val_acc = history.history['val_accuracy'][-1]
    print(f"lr={hparams['lr']}, dense={hparams['dense_units']}, drop={hparams['dropout']:.2f} --> val_acc={val_acc:.4f}")
    
    return {'loss': -val_acc, 'status': STATUS_OK}

print("--- Hyperparameter Optimization --- ")

space = {
    'lr': hp.choice('lr', [1e-3, 1e-4]),
    'dense_units': hp.choice('dense_units', [64, 128, 256]),
    'dropout': hp.uniform('dropout', 0.25, 0.5),
}

trials = Trials()
best = fmin(
    fn=build_and_train,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

print("Best hyperparameters:", best)
with open('best_hyperparameters.txt', 'w') as f:
    for key, value in best.items():
        f.write(f"{key}: {value}\n")
print("Hyperparameter optimization completed.")