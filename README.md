# Plant Disease Classification

This project classifies plant diseases using image data.

## Prerequisites

* Python 3.x
* pip
* Git

## Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/MeeraLizJoy/PlantDiseaseXGBoost.git
    cd <project_directory>
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Download the PlantifyDr dataset and place it in the `data` directory. The dataset should have `train` and `validation` subdirectories.

## Running the Project

1.  Run the main script:

    ```bash
    python src/main.py
    ```

## Notes

* The project uses EfficientNet for feature extraction and XGBoost for classification.
* Memory usage can be high. If you encounter memory issues, try reducing the batch size or using PCA to reduce feature dimensionality.
* The project uses caching to speed up feature extraction.
* The model will be saved as `data/xgb_model.pkl`.
* The feature extraction files will be saved in `data/`.

## Author

* Your Name# PlantDiseaseXGBoost
