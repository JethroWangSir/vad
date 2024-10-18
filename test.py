import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, fbeta_score
from tqdm import tqdm
import time

from dataset import AVA
from sgvad import SGVAD

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experiment directory
name = 'exp_sgvad'
exp_dir = f'./exp/{name}/'
os.makedirs(exp_dir, exist_ok=True)

# Load test dataset
test_dataset = AVA('/share/nas165/aaronelyu/Datasets/AVA-speech/')
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize SGVAD model and load the best checkpoint
sgvad = SGVAD.init_from_ckpt()

# Variables for storing evaluation metrics and time
all_labels = []
all_preds = []
inference_times = []  # To store inference times for each batch

# Test loop
with torch.no_grad():
    for audio_path, label in tqdm(test_loader, desc='Testing'):
        audio_path, label = audio_path.to(device), label.to(device)

        # Measure inference time
        start_time = time.time()  # Start time before prediction
        prediction = sgvad.predict(audio_path)  # Predict using SGVAD
        end_time = time.time()  # End time after prediction

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)  # Store the inference time

        # Record predictions and true labels
        all_preds.append(prediction)
        all_labels.append(label)

# Calculate average inference time in milliseconds
average_inference_time = sum(inference_times) / len(inference_times)

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Compute evaluation metrics
auroc = roc_auc_score(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
f2_score = fbeta_score(all_labels, all_preds, beta=2)

# Calculate confusion matrix and derive FPR, FNR
conf_matrix = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = conf_matrix.ravel()  # Extract true negatives, false positives, false negatives, and true positives
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

# Save results to JSON
results = {
    "AUROC": auroc,
    "Accuracy": accuracy,
    "F2-Score": f2_score,
    "FPR": fpr,
    "FNR": fnr,
    "Average Inference Time (ms)": average_inference_time,
}

# Save results as JSON
results_file = os.path.join(exp_dir, 'test.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

# Print metrics to console
print(f"AUROC: {auroc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F2-Score: {f2_score:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")
print(f"Average Inference Time: {average_inference_time:.4f} ms")
