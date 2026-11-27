:::writing{variant="document" id="73184"}
# CardioSignal – Model Card

## 1. Model Overview

**Model name:** CardioSignal ResNet1D  
**Model type:** One-dimensional Residual Convolutional Neural Network (ResNet1D)  
**Task:** Multi-label classification of 12-lead electrocardiogram (ECG) signals  
**Framework:** PyTorch  
**Dataset:** PTB-XL v1.0.3  
**Status:** Experimental research model

CardioSignal ResNet1D is a deep learning model designed to classify ECG recordings into five diagnostic superclasses. It processes 12-lead ECG signals and produces independent prediction scores for each configured class.

The model is part of an end-to-end ECG classification pipeline that includes data loading, preprocessing, training, checkpoint management, and evaluation.

## 2. Intended Use

### Primary Intended Use

The model is intended for:

- Educational and research purposes.
- Experimentation with deep learning for ECG signal classification.
- Benchmarking multi-label classification methods on PTB-XL.
- Exploring one-dimensional residual neural network architectures for time-series data.

### Out-of-Scope Uses

The model is not intended for:

- Clinical diagnosis or medical decision-making.
- Replacing cardiologists or other qualified healthcare professionals.
- Emergency medical assessment.
- Autonomous patient monitoring or treatment recommendations.
- Deployment in clinical environments without appropriate validation, regulatory assessment, and clinical oversight.

**Important:** This model is an experimental research implementation, not a medical device. Its predictions must not be interpreted as clinical diagnoses.

## 3. Diagnostic Classes

The model predicts five PTB-XL diagnostic superclasses:

| Label | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| STTC | ST/T change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

The task is multi-label classification. An ECG recording can have multiple positive labels simultaneously.

The labels are derived by mapping PTB-XL SCP-ECG statements to the configured diagnostic superclasses.

## 4. Model Inputs and Outputs

### Input

| Property | Value |
|---|---|
| Signal type | 12-lead ECG |
| Sampling frequency | 500 Hz |
| Recording duration | 10 seconds |
| Samples per lead | 5,000 |
| Input shape | `(batch_size, 12, 5000)` |
| Input data type | float32 |

### Output

The model produces five raw logits, one for each diagnostic superclass.

Sigmoid is applied to convert logits into independent class probabilities.

The configured classification threshold is `0.5`.

A probability equal to or greater than the threshold is classified as a positive prediction for that class.

## 5. Training Data

### Dataset Source

The model was trained using:

**PTB-XL v1.0.3**

- Source: https://physionet.org/content/ptb-xl/1.0.3/
- DOI: https://doi.org/10.13026/kfzx-aw45
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)

PTB-XL provides 10-second, 12-lead ECG recordings.

The project uses the 500 Hz recordings from `records500`.

### Dataset Splitting

The official patient-wise folds were used:

| Split | PTB-XL Folds | ECG Records |
|---|---|---:|
| Training | 1–8 | 17,084 |
| Validation | 9 | 2,146 |
| Test | 10 | 2,158 |
| **Total** | | **21,388** |

Records without usable target labels were excluded.

Patient-level disjointness was verified between training, validation, and test splits to prevent patient overlap across the three subsets.

### Class Distribution

The number of positive labels in each split was:

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| NORM | 7,596 | 955 | 963 |
| MI | 4,379 | 540 | 550 |
| STTC | 4,186 | 528 | 521 |
| CD | 3,907 | 495 | 496 |
| HYP | 2,119 | 268 | 262 |

Because this is a multi-label task, a single recording may contribute to more than one class count.

## 6. Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads high-resolution ECG recordings at 500 Hz.
2. Maps SCP-ECG statements to five diagnostic superclasses.
3. Excludes records without usable labels.
4. Uses the official patient-wise PTB-XL folds.
5. Validates ECG dimensions, lead order, and signal values.
6. Applies per-lead z-score normalization.

Normalization statistics were computed using the training split only.

For each ECG lead, the training-set mean and standard deviation were calculated and used to normalize the training, validation, and test signals.

This prevents normalization statistics from being calculated using the validation or test data.

## 7. Model Architecture

CardioSignal uses a one-dimensional ResNet architecture.

The architecture is designed to process ECG time-series signals using convolutional layers and residual connections.

The model:

- Accepts 12-channel ECG input.
- Uses configurable residual stages and channel widths.
- Produces five output logits.
- Supports multi-label classification.

The implementation is available in:

`src/cardiosignal/models/resnet1d.py`

The training pipeline also includes a separate configurable CNN1D implementation. The results in this model card refer specifically to ResNet1D.

## 8. Training Configuration

The selected ResNet1D model was trained using:

| Parameter | Configuration |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss function | BCEWithLogitsLoss |
| Batch size | 32 |
| Maximum epochs | 50 |
| Completed epochs | 32 |
| Random seed | 42 |
| Learning rate scheduler | ReduceLROnPlateau |
| Early stopping patience | 10 |
| Best checkpoint epoch | 22 |

Training stopped early when validation macro F1 did not improve according to the configured early-stopping criteria.

The best checkpoint was selected using validation macro F1.

### Best Validation Result

| Metric | Score |
|---|---:|
| Best validation macro F1 | 0.74779 |
| Best epoch | 22 |

## 9. Evaluation Results

The selected checkpoint was evaluated on the held-out PTB-XL test set.

**Test samples:** 2,158 ECG recordings  
**Classification threshold:** 0.5

### Overall Performance

| Metric | Test Score |
|---|---:|
| Macro F1 | 0.75474 |
| Micro F1 | 0.78962 |
| Macro AUROC | 0.93206 |
| Macro AUPRC | 0.83157 |
| Test Loss | 0.26717 |

### Per-Class Performance

| Class | F1 | AUROC | AUPRC |
|---|---:|---:|---:|
| NORM | 0.8695 | 0.9498 | 0.9339 |
| MI | 0.7537 | 0.9381 | 0.8626 |
| STTC | 0.7697 | 0.9351 | 0.8299 |
| CD | 0.7581 | 0.9254 | 0.8533 |
| HYP | 0.6227 | 0.9120 | 0.6782 |

### Confusion Matrix Results

Each diagnostic superclass was evaluated independently using a binary confusion matrix.

| Class | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| NORM | 1,019 | 176 | 87 | 876 |
| MI | 1,523 | 85 | 166 | 384 |
| STTC | 1,517 | 120 | 120 | 401 |
| CD | 1,565 | 97 | 134 | 362 |
| HYP | 1,855 | 41 | 125 | 137 |

## 10. Performance Analysis

The evaluation results show different levels of classification performance across the five diagnostic classes.

- NORM achieved the highest F1 score (0.8695).
- MI, STTC, and CD achieved F1 scores between approximately 0.75 and 0.77.
- HYP was the most challenging class, with an F1 score of 0.6227.
- HYP recall was approximately 0.5229, indicating that a substantial number of positive HYP cases were not detected at the configured threshold.

These results identify opportunities for further experimentation, particularly around class-specific performance, threshold selection, and model architecture.

## 11. Limitations and Potential Biases

Several limitations should be considered when interpreting the results.

### Dataset Scope

The model was trained and evaluated using PTB-XL v1.0.3. Its performance on this dataset does not establish performance on other hospitals, patient populations, ECG devices, or recording protocols.

### Class Imbalance

The five diagnostic classes have different numbers of positive examples. HYP has fewer positive test examples than NORM, MI, STTC, and CD.

This difference may be relevant when interpreting class-specific performance, but the current experiment does not establish a causal explanation for the lower HYP recall.

### Classification Threshold

The evaluation used a fixed threshold of 0.5 for all classes. Class-specific threshold optimization and probability calibration have not been established in this experiment.

### External Validation

No independent external dataset evaluation has been completed.

### Clinical Validation

The model has not undergone clinical validation, prospective evaluation, or assessment by healthcare professionals.

### Fairness and Subgroup Evaluation

Demographic subgroup performance and fairness across patient groups have not been evaluated in this project.

## 12. Ethical Considerations and Responsible Use

ECG classification is a health-related application. Incorrect predictions may have serious consequences if interpreted as medical advice.

The model should only be used for educational and experimental research.

Predictions should not be used to diagnose patients, guide treatment, or replace professional medical interpretation.

The project does not claim clinical effectiveness or readiness for healthcare deployment.

## 13. Reproducibility

The project uses:

- Random seed: 42
- YAML-based configuration files
- Official patient-wise dataset folds
- Training-only normalization statistics
- Saved model checkpoints
- Separate training, validation, and test workflows

The best ResNet1D checkpoint was saved during training and used for final test evaluation.

Exact reproducibility may depend on the PyTorch version, hardware, CUDA environment, and numerical behavior of accelerator operations.

## 14. Model Artifacts

The trained checkpoint and evaluation reports are generated locally and are not included in the Git repository.

The evaluation outputs include:

- Test metrics in JSON format.
- Per-class metrics in CSV format.
- Test metrics comparison plot.
- Per-class confusion matrix visualizations.

Generated evaluation figures are documented in the project README.

## 15. Future Work

Potential future improvements include:

- Benchmarking CNN1D against ResNet1D using the same data splits.
- Investigating class-specific classification thresholds.
- Exploring strategies for handling class imbalance.
- Performing additional model and hyperparameter experiments.
- Evaluating the model on independent ECG datasets.
- Improving interpretability and model analysis.
- Conducting appropriate external and clinical validation before considering any healthcare application.

## 16. References

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). *PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45

Wagner, P., et al. (2020). PTB-XL: A Large Publicly Available ECG Dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6
:::
