# visualization_tasks.py
# This module contains all functions related to creating plots from the report.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

def create_all_visuals(csv_path, output_dir):
    """
    The main function to generate and save all visualizations from a report file.
    """
    print("\n--- Starting Visualization Generation Task ---")
    
    # Check if the report file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: Report file not found at '{csv_path}'.")
        print("Cannot generate visualizations. Please run the analysis first.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded report with {len(df)} entries.")
    
    # --- Generate all plots ---
    _plot_confusion_matrix(df, os.path.join(output_dir, 'confusion_matrix.png'))
    _plot_confidence_histogram(df, os.path.join(output_dir, 'confidence_histogram.png'))
    _plot_roc_curve(df, os.path.join(output_dir, 'roc_curve.png'))
    
    print("\n--- All visualizations have been saved to the '{}' folder. ---".format(output_dir))


# --- Helper functions (prefixed with _ to indicate internal use) ---

def _plot_confidence_histogram(df, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Confidence', hue='True Class', multiple='stack', bins=30, kde=True)
    plt.title('Distribution of Model Confidence Scores by True Class')
    plt.xlabel('Confidence Score (Probability of being Class 1)')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()

def _plot_confusion_matrix(df, save_path):
    cm = confusion_matrix(df['True Class'], df['Predicted Class'])
    class_names = ['not_center (0)', 'center (1)']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual (True) Class')
    plt.xlabel('Predicted Class')
    plt.savefig(save_path)
    plt.close()

def _plot_roc_curve(df, save_path):
    fpr, tpr, _ = roc_curve(df['True Class'], df['Confidence'])
    auc_score = roc_auc_score(df['True Class'], df['Confidence'])
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()