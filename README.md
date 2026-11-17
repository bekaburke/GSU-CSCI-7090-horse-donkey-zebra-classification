# Horse, Donkey, and Zebra Image Classification

**Georgia Southern University | CSCI 7090 – Advanced Computer Vision**

## Project Overview

This project explores animal species classification using convolutional neural networks (CNNs) and transfer learning.

The objective was to classify images into three categories: **horse, donkey, and zebra**, using pretrained deep learning architectures and comparing their performance under different training strategies.

The project investigated whether increased model complexity, fine-tuning, and image augmentation improve classification performance on a relatively small dataset.

## Dataset

The dataset was obtained from Kaggle and contains 1,265 images across three animal classes.

| Dataset Split | Images |
|---|---:|
| Training | 807 |
| Validation | 202 |
| Original test set | 256 |
| Corrected test set | 255 |

All images have dimensions of 224 × 224 pixels.

The original training directory was divided into training and validation subsets using an 80/20 stratified split. The provided test directory was kept separate for final evaluation.

### Data Quality

An image-quality audit identified potential problems including illustrations, embedded text, unusual backgrounds, and suspected duplicates.

SHA-256 hashing confirmed one exact duplicate between the training and test datasets. The duplicate was excluded from the corrected test evaluation.

Additional manually identified duplicate candidates require verification.

## Technologies

- Python
- TensorFlow / Keras
- Google Colab
- ResNet152V2
- MobileNetV2
- OpenCV
- NumPy and Pandas
- Scikit-learn
- Matplotlib

## Methodology

### 1. Preprocessing

Images were prepared using architecture-specific preprocessing functions. ResNet152V2 preprocessing scales pixel values to approximately -1 to 1.

Numeric class labels were assigned to donkey, horse, and zebra images.

### 2. Transfer Learning

Two pretrained ImageNet architectures were evaluated:

- **ResNet152V2:** A deep residual network used as the primary classification model.
- **MobileNetV2:** A lightweight architecture evaluated for computational efficiency and classification performance.

Both models were initially trained with their pretrained feature-extraction layers frozen and a new three-class classification head.

### 3. Fine-Tuning

The final 20 layers of the ResNet152V2 backbone were selectively unfrozen, while batch-normalization layers remained frozen.

The fine-tuned model was trained using a reduced learning rate of 0.00001.

### 4. Early Stopping

Upon professor recommendation, early stopping was implemented for the frozen ResNet152V2, fine-tuned ResNet152V2, and MobileNetV2 experiments.

Training monitored validation loss and stopped after three consecutive epochs without sufficient improvement, restoring the best-performing weights.

### 5. Data Augmentation

A separate ResNet152V2 experiment evaluated training-only augmentation, including:

- Horizontal flipping
- Mild rotation
- Mild zoom
- Brightness adjustments

Validation and test images were not augmented.

## Experimental Results

| Model | Best Validation Accuracy |
|---|---:|
| Frozen ResNet152V2 | **96.04%** |
| Fine-tuned ResNet152V2 | 95.05% |
| MobileNetV2 | 95.54% |
| Augmented ResNet152V2 | 95.05% |

### Final Test Evaluation

The frozen ResNet152V2 model was selected based on validation performance.

After excluding one confirmed training/test duplicate:

- **Corrected test accuracy: 92.94%**
- Correct predictions: 237 of 255
- Most classification errors occurred between horses and donkeys.
- Zebra classification demonstrated the highest recall.

## Key Findings

1. **Model complexity:** ResNet152V2 achieved the highest validation accuracy, but MobileNetV2 performed within 0.50 percentage points while using approximately 96% fewer parameters.

2. **Fine-tuning:** Unfreezing additional ResNet152V2 layers did not improve validation accuracy in this experiment.

3. **Early stopping:** Validation-loss monitoring prevented unnecessary additional training and restored model weights from the best-performing epochs.

4. **Augmentation:** The augmented ResNet152V2 experiment did not improve validation accuracy compared with the frozen baseline.

5. **Data quality:** The identification of a train/test duplicate demonstrated the importance of auditing datasets before interpreting evaluation results.

## Repository Structure

- `Final_DHZ_Dataset_Work (1).ipynb` — Complete Google Colab notebook containing preprocessing, training, evaluation, and visualizations.
- `results/` — Training histories, accuracy/loss plots, confusion matrices, and comparison data.
- `presentation/` — Final project presentation.

## Reproducing the Project

1. Download the original Horse, Donkey, and Zebra classification dataset from Kaggle.
2. Organize the images into `train` and `test` directories, each containing `horse`, `donkey`, and `zebra` subdirectories.
3. Open the notebook in Google Colab.
4. Enable a GPU runtime.
5. Mount Google Drive and update dataset paths to match your environment.
6. Execute the notebook cells sequentially.

The notebook uses pretrained ImageNet weights downloaded through TensorFlow/Keras. The original dataset and trained model files are not included in this repository.

## Limitations and Future Work

The dataset is relatively small and contains class imbalance and image-quality inconsistencies.

Potential background-based shortcuts, labeling problems, and suspected duplicate images warrant further investigation.

Future work could include additional dataset cleaning, systematic error analysis, evaluation on independently collected images, and exploration of alternative augmentation and class-balancing strategies.

## Author

Graduate student, M.S. Computer Science  
Georgia Southern University

## Dataset Attribution

Original dataset: *Horse, Donkey, and Zebra Image Classification*, available on Kaggle.

(https://www.kaggle.com/datasets/ifeanyinneji/donkeys-horses-zebra-images-dataset)
