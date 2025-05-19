# CNN Spectrogram Emotion Recognition

This project implements a Convolutional Neural Network (CNN) to classify emotions from spectrogram images generated from audio recordings. The Jupyter Notebook `cnn_spectogram_emotion_recogntion_final.ipynb` contains the complete workflow from data loading and preprocessing to model training and evaluation.

## Features

- **Data Handling**: Supports loading and preprocessing audio data from multiple emotion speech datasets:
    - CREMA-D
    - SAVEE
    - TESS
- **Spectrogram Generation**: Converts audio files into Mel spectrograms and saves them as images, which serve as input to the CNN.
- **CNN Models**: Implements and experiments with various CNN architectures:
    - A simple 3-layer CNN.
    - An adapted multi-branch CNN.
    - Model variations incorporating data augmentation techniques (e.g., resizing, random rotation, random flip).
    - A model using a weighted categorical cross-entropy loss function to address class imbalance, particularly for 'fear' and 'happy' emotions.
- **Training and Evaluation**:
    - Trains the defined CNN models on the prepared spectrogram dataset.
    - Evaluates model performance using standard classification metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-Score (derived from precision and recall)
    - Visualizes training progress (loss, accuracy, precision, recall over epochs).
    - Displays confusion matrices to analyze classification performance across different emotions (angry, fear, happy, sad).
- **Dependencies**: The notebook utilizes common Python libraries for machine learning and data processing, including:
    - TensorFlow/Keras for building and training neural networks.
    - Librosa for audio processing and spectrogram generation.
    - Pandas for data manipulation.
    - Matplotlib and Scikit-learn for plotting and evaluation.
    - `visualkeras` for visualizing model architectures.
    - `patool` for archive extraction.

## Workflow

1.  **Setup**: Installs necessary libraries like `patool` and `visualkeras`. Mounts Google Drive (if running in Colab) to access datasets.
2.  **Data Loading and Preprocessing**:
    - Extracts datasets (e.g., from a `.rar` archive).
    - Defines functions (`load_crema_dataset`, `load_savee_dataset`, `load_tess_dataset`) to load audio file paths and their corresponding emotion labels into Pandas DataFrames.
    - Defines a function (`save_mel_spectrogram_as_image`) to convert audio signals to log Mel spectrograms and save them as PNG images.
    - Loads the pre-generated spectrogram images (`X`) and their corresponding labels (`y`).
    - Normalizes image pixel values and performs one-hot encoding on labels.
    - Splits the data into training, validation, and test sets.
3.  **Model Definition**:
    - Several CNN model architectures are defined:
        - `create_3layer_model`
        - `build_adapted_cnn`
        - `create_3layer_separable_conv_model`
        - `create_dscnn_model`
        - `create_3layer_model_with_augmentation`
        - `build_adapted_cnn_with_augmentation`
        - `build_adapted_cnn_with_augmentation_weighted` (includes data augmentation and a custom weighted loss function).
    - Helper functions like `plot_history` and `display_conf` are provided for visualizing training results and confusion matrices.
4.  **Model Training and Evaluation**:
    - An instance of one of the defined models is created (e.g., `model1 = create_3layer_model(...)` or an adapted CNN).
    - The model is compiled with an optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy or weighted categorical cross-entropy), and metrics.
    - The model is trained using the training data (`X_train`, `y_train`) and validated on the validation set (`X_val`, `y_val`).
    - Early stopping is used as a callback to prevent overfitting.
    - After training, the model's performance is evaluated on the test set.
    - Training history plots and confusion matrices are displayed to assess the model.

## Usage

1.  Ensure all dependencies listed in the notebook are installed.
2.  Make sure the dataset paths are correctly configured. The notebook expects spectrogram images in a directory specified by `datapath` and labels in a `.npy` file.
3.  Run the cells in the Jupyter Notebook sequentially.
4.  Experiment with different model architectures, hyperparameters, and data augmentation techniques to potentially improve performance.

## Emotion Classes

The models are trained to classify the following emotions:
- Angry
- Fear
- Happy
- Sad

The notebook also includes initial data loading functions that can handle 'disgust' and 'neutral' emotions, but the final classification task focuses on the four classes above. 