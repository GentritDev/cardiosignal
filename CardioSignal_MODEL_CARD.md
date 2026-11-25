# CardioSignal — Model Card

> **Status:** Experimental project. Full model training and full-split evaluation are pending.  
> **Intended use:** Research, learning, and software-engineering demonstration only.  
> **Not for clinical use:** This model is not a medical device and must not be used to diagnose, screen, triage, or make treatment decisions.

## 1. Model overview

CardioSignal is an experimental deep-learning pipeline for multi-label classification of 12-lead electrocardiogram (ECG) recordings. It is designed to compare one-dimensional convolutional neural network architectures on the PTB-XL dataset.

The configured target labels are:

| Label | Meaning in the PTB-XL diagnostic-superclass task |
|---|---|
| `NORM` | Normal ECG |
| `MI` | Myocardial infarction |
| `STTC` | ST/T change |
| `CD` | Conduction disturbance |
| `HYP` | Hypertrophy |

The task is **multi-label classification**: more than one label may apply to a recording. The model outputs raw logits; sigmoid activation is used to obtain per-class scores for inference. The configured decision threshold is `0.5`.

## 2. Model details

- **Project:** CardioSignal
- **Model family:** 1D convolutional neural networks for time-series signals
- **Configured default architecture:** ResNet1D
- **Alternative architecture:** CNN1D
- **Input channels:** 12 ECG leads
- **Configured input length:** 5,000 samples per lead
- **Sampling rate:** 500 Hz
- **Recording duration:** 10 seconds
- **Output classes:** 5
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** AdamW
- **Configured learning rate:** `0.001`
- **Configured weight decay:** `0.0001`
- **Configured maximum epochs:** 50, with early stopping enabled
- **Configured batch size:** 32
- **Random seed:** 42

The ResNet1D configuration uses channel widths `[32, 64, 128, 256]`, two residual blocks per stage, and dropout `0.3`. The CNN1D configuration uses channel widths `[32, 64, 128, 256]` and dropout `0.3`.

These are configuration values, not a claim that a final model has completed training with these settings.

## 3. Training data

- **Dataset:** PTB-XL, version 1.0.3
- **Configured record resolution:** 500 Hz (`records500`)
- **Signal shape:** 12 leads × 5,000 samples
- **Metadata:** `ptbxl_database.csv`
- **Diagnostic statements:** `scp_statements.csv`
- **Unlabeled records:** Excluded by the current configuration

The project uses the official PTB-XL fold assignment configured as follows:

| Split | Folds | Number of records in the current local data preparation |
|---|---:|---:|
| Train | 1–8 | 17,084 |
| Validation | 9 | 2,146 |
| Test | 10 | 2,158 |

The counts above reflect the current project data-preparation run. They should be revalidated whenever the dataset version, filtering rules, or preprocessing pipeline changes.

## 4. Preprocessing

The current configuration specifies:

1. Read 12-lead ECG recordings at 500 Hz.
2. Use 10-second recordings represented by 5,000 samples per lead.
3. Convert the input layout from time-by-channel to channel-by-time for model input.
4. Apply per-lead z-score normalization using statistics computed from the training split only.
5. Reject non-finite values.

Normalization statistics are configured to be saved to `artifacts/normalization_stats.json`. Inference must use the same training-derived statistics as the corresponding trained checkpoint; statistics must not be recomputed from validation, test, or individual inference records.

## 5. Evaluation

The training configuration requests the following evaluation metrics:

- Macro F1
- Micro F1
- Per-class F1
- Macro AUROC
- Macro AUPRC

The validation and test sets must remain separate from training. The test split should be used for final reporting only after model selection and configuration decisions have been completed using the training and validation splits.

### Current evaluation status

**Final performance results are not yet available.** The project has passed its current automated test suite (49 tests) and has completed small smoke-training and smoke-evaluation runs to verify that the pipeline executes. Those smoke runs used only a tiny subset of batches and are not representative estimates of generalization performance. Their metrics must not be reported as final model performance.

When full training is complete, add a results table here containing the checkpoint identifier, evaluation split, number of evaluated records, thresholding method, aggregate metrics, and per-class metrics. Report test results only for the final selected model.

## 6. Intended use

Appropriate uses include:

- Educational exploration of ECG time-series classification.
- Experimentation with 1D CNN and ResNet1D architectures.
- Demonstration of data loading, preprocessing, training, evaluation, experiment tracking, and model-serving workflows.
- Reproducible research experiments on the PTB-XL dataset, subject to its terms of use.

## 7. Out-of-scope and prohibited uses

This project is **not validated for clinical use**. It must not be used as the basis for:

- Diagnosing or ruling out a cardiac condition.
- Patient screening, triage, monitoring, or treatment decisions.
- Replacing a qualified clinician or an approved medical device.
- Making decisions about an individual's access to healthcare, insurance, employment, or other services.

Predicted scores are model outputs, not calibrated clinical probabilities unless calibration is separately evaluated and documented.

## 8. Limitations and risks

- The project is experimental and has no completed full-split performance report yet.
- Performance may vary across hospitals, ECG devices, patient populations, acquisition protocols, and demographic groups.
- The five diagnostic superclasses do not represent every possible cardiac condition or ECG finding.
- Multi-label thresholds can affect sensitivity and precision; the default threshold of `0.5` has not been established as clinically appropriate.
- Dataset labels may contain noise, ambiguity, and class imbalance.
- Training and validation on PTB-XL do not establish real-world clinical validity or external generalization.
- A model may learn dataset-specific or acquisition-related patterns rather than clinically meaningful signal features.

## 9. Ethical considerations

ECG data is health-related data. Anyone using this project must follow the dataset's access conditions, applicable privacy and data-protection requirements, and institutional review requirements where relevant. Do not upload identifiable patient data, secrets, credentials, or private clinical records to the repository or public services.

Potential downstream use requires appropriate clinical, privacy, security, fairness, and regulatory review. Evaluation should include subgroup analysis and external validation before any consideration of a clinical setting.

## 10. Reproducibility and artifacts

The repository contains configuration files under `configs/` and source code under `src/cardiosignal/`. The current configured seed is `42`. Training checkpoints, normalization statistics, local MLflow data, and dataset files are intentionally excluded from Git and must be managed separately.

For a reproducible experiment, record at minimum:

- Git commit hash.
- Dataset version and exact data filtering rules.
- Python, PyTorch, CUDA, and dependency versions.
- Model architecture and all hyperparameters.
- Training/validation split definitions.
- Checkpoint hash or identifier.
- Normalization-statistics artifact identifier.
- Evaluation split, threshold, and complete metrics.

## 11. Contact and project status

This is an evolving portfolio and research project. Update this model card whenever the architecture, data, preprocessing, training procedure, evaluation results, or intended use changes.
