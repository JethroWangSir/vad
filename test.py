import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset import AVA
from sgvad import SGVAD

# Load test dataset
test_dataset = AVA('/mnt/500g/vad/dataset/AVA/AVA-speech/')
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize SGVAD model and load the best checkpoint
sgvad = SGVAD.init_from_ckpt()

# Variables for storing evaluation metrics and time
all_labels = []
all_preds = []

# Test loop
with torch.no_grad():
    for audio_path, label in tqdm(test_loader, desc='Testing'):
        audio_path, label = audio_path[0], label.item()

        # Predict using SGVAD
        prediction = sgvad.predict(audio_path)

        # Record predictions and true labels
        all_preds.append(prediction)
        all_labels.append(label)

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Compute evaluation metrics
auroc = roc_auc_score(all_labels, all_preds)

# Save results to JSON
results = {"AUROC": auroc}
results_file = 'test.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

# Print metrics to console
print(f"AUROC: {auroc:.4f}")
