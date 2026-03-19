# iCNN-LSTM+ Ransomware Detection (Research Reproduction Project)

This project reproduces the deep learning section of the iCNN-LSTM+ ransomware detection paper using Sysmon-derived SILRAD features, with a full batch-incremental learning workflow and presentation-ready reporting.

## Project Summary

Goal:
Build a research-grade ransomware classifier that can be continuously updated on streaming mini-batches without full retraining.

Core ideas implemented:

1. Hybrid CNN-LSTM+attention architecture for spatial and temporal pattern learning.
2. Batch-based incremental learning (initial baseline + periodic updates).
3. Recall-focused evaluation using F2 score for ransomware detection.
4. Automated result packaging: metrics tables, visual plots, and HTML dashboard.
<img width="767" height="737" alt="image" src="https://github.com/user-attachments/assets/0c09ae5b-e06e-45d1-a66d-acc20ef0f22e" />



## Implemented Architecture

Input:

1. 36 numerical Sysmon-derived features per event.

Model pipeline:

1. Sequential 1D CNN block.
2. Parallel stacked LSTM branches.
3. Attention mechanism per branch.
4. Concatenation + dense classifier head.

Incremental learning pipeline:

1. Initial model build on first 40,000 events.
2. Stream updates in windows of 10,000 events.
3. Per update window split into 80 percent train and 20 percent validation.
4. SMOTE-enabled class balancing in training splits.
5. Stage-aware training logs to avoid confusion with epoch resets across batches.
<img width="922" height="548" alt="image" src="https://github.com/user-attachments/assets/cba793b9-6f73-46fb-a05f-46ca1f7d7c52" />


## Hyperparameters (Table 4 Applied)

1. CNN kernel size: 9
2. CNN filters: 32
3. LSTM units: 384
4. Dropout: 0.10326648213511579
5. Activation: tanh (hidden), sigmoid (output)
6. Optimizer: Adam
7. Learning rate: 0.001
8. Dense layers: 80, 2
9. Epochs: 100
10. Batch size in paper: 1024


## Reproducibility and Environment

Local Python environment:

1. Python virtual environment in workspace.
2. Dependencies in requirements.txt.

Recommended GPU run path (used in this project):

1. Docker Desktop + NVIDIA driver.
2. TensorFlow 2.17 GPU container.
3. Mounted workspace for direct artifact generation.

## How to Run

### A) Local run (CPU)

powershell
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
run_train.py --output-dir outputs src.icnn_lstm.report --metrics-json outputs/incremental_metrics.json --output-dir outputs/reports

### B) Docker GPU run (recommended)

powershell
docker run --rm --gpus all -e TF_FORCE_GPU_ALLOW_GROWTH=true -v "${PWD}:/workspace" -w /workspace tensorflow/tensorflow:2.17.0-gpu bash -lc "python -m pip install --no-input pandas scikit-learn imbalanced-learn matplotlib pyyaml && python run_train.py --initial-events 40000 --update-events 10000 --initial-epochs 100 --update-epochs 100 --fit-batch-size 64 --output-dir outputs && python -m src.icnn_lstm.report --metrics-json outputs/incremental_metrics.json --output-dir outputs/reports"

## Project Structure

1. run_train.py: training entrypoint.
2. src/icnn_lstm/model.py: iCNN-LSTM+ architecture.
3. src/icnn_lstm/train_incremental.py: staged baseline + incremental updates.
4. src/icnn_lstm/data.py: loading, scaling, split logic, SMOTE.
5. src/icnn_lstm/metrics.py: accuracy, precision, recall, F2.
6. src/icnn_lstm/report.py: report generator (CSV + plots + dashboard).
7. outputs/reports: generated presentation artifacts.

## Latest Experiment Snapshot (GPU Run)

Source: outputs/reports/run_summary.json

1. Incremental batches: 15
2. Initial training time: 319.16 seconds
3. Total runtime: 1692.52 seconds
4. Mean F2 score: 0.8776
5. Median F2 score: 0.9881
6. Mean recall: 0.9485
7. Mean precision: 0.7764
8. Mean accuracy: 0.9644
9. Best batch F2: 1.0000
10. Worst batch F2: 0.4351
<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/3d597291-5435-497d-a4d7-9aec0c022240" />



Interpretation:
The model maintains very high recall on average, which is critical in ransomware detection where false negatives are high-risk.

## Presentation Artifacts (Professor/Interview Ready)

Generated at outputs/reports:

1. run_summary.json: compact KPI summary.
2. metrics_by_batch.csv: per-batch quantitative results.
3. metrics_over_batches.png: F2/Recall/Precision trend plot.
4. batch_runtime_seconds.png: update-time profile across batches.
5. dashboard.html: single-page visual dashboard.
6. report.md: narrative report.


## Limitations and Future Work

1. Further tuning can improve robustness on low-ransomware-ratio batches.
2. Potential improvements: drift-aware scheduling, threshold calibration, and confidence-based retraining triggers.

## Reference

Primary paper implemented:
iCNN-LSTM+: A Batch-Based Incremental Ransomware Detection System Using Sysmon.
