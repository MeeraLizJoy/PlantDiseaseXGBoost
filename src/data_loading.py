# src/data_loading.py
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from memory_profiler import profile

@profile
def create_data_generators(data_dir, img_width, img_height, batch_size=16):
    """Creates ImageDataGenerators for training and validation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = valid_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    print(f"Found {train_generator.samples} images in training set.")
    print(f"Found {validation_generator.samples} images in validation set.")

    return train_generator, validation_generator, train_generator.class_indices

@profile
def preprocess_labels(class_indices):
    """Creates a LabelEncoder from class_indices."""
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_indices.keys()))
    print(f"Label encoder classes: {label_encoder.classes_}")
    return label_encoder

@profile
def batch_generator(generator, feature_extractor, cache_file=None):
    """Generates batches of features and labels."""
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_features = feature_extractor.predict(batch_x)
        if cache_file:
            if i == 0:
                if os.path.exists(cache_file):
                    print (f"Loading cache from: {cache_file}")
                    yield np.load(cache_file), batch_y
                else:
                    print (f"Creating cache: {cache_file}")
                    np.save(cache_file, batch_features)
                    yield batch_features, batch_y
            else:
                loaded_cache = np.load(cache_file)
                batch_features = np.concatenate((loaded_cache, batch_features), axis = 0)
                np.save(cache_file, batch_features)
                yield batch_features, batch_y
        else:
            yield batch_features, batch_y