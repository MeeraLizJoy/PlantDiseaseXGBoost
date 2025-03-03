# src/feature_extraction.py
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os
from memory_profiler import profile

@profile
def create_feature_extractor(img_width, img_height, efficientnet_version='B0'):
    """Creates feature extractor with specified EfficientNet version."""
    if efficientnet_version == 'B0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    elif efficientnet_version == 'B1':
        from tensorflow.keras.applications import EfficientNetB1
        base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    elif efficientnet_version == 'B3':
        from tensorflow.keras.applications import EfficientNetB3
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    else:
        raise ValueError("Invalid EfficientNet version. Choose 'B0', 'B1', or 'B3'.")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

@profile
def extract_features(images, feature_extractor, cache_file=None):
    """Extracts features and caches them if a cache file is provided."""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return np.load(cache_file)

    processed_images = preprocess_input(images)
    features = feature_extractor.predict(processed_images)

    if cache_file:
        print(f"Caching features to {cache_file}")
        np.save(cache_file, features)

    return features