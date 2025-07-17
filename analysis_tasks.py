# analysis_tasks.py (Updated)
import os
import csv
import math
from prediction_utils import get_prediction_details

# Import metrics from scikit-learn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def calculate_log_loss(true_label, predicted_prob, epsilon=1e-15):
    """
    Calculates the Log Loss for a single prediction.
    """
    predicted_prob = max(epsilon, min(1 - epsilon, predicted_prob))
    if true_label == 1:
        return -math.log(predicted_prob)
    else:
        return -math.log(1 - predicted_prob)

def print_summary_metrics(true_labels, predicted_labels):
    """
    Calculates and prints summary classification metrics with a robust confusion matrix.
    """
    print("\n--- Overall Performance Metrics ---")
    
    # Calculate standard metrics
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    # --- THIS IS THE ROBUST FIX ---
    # Create the confusion matrix. We specify labels=[0, 1] to ensure
    # the matrix is always 2x2, even if one class is missing in the data.
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    
    # Now we can safely extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    # -----------------------------
    
    # Print metrics
    print(f"F1 Score:         {f1:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"Actual 0         {tn:<13} {fp:<13}")
    print(f"Actual 1         {fn:<13} {tp:<13}\n")
    print(f"  - True Negatives (TN):  {tn} (Correctly predicted 'not_center')")
    print(f"  - False Positives (FP): {fp} (Incorrectly predicted 'center')")
    print(f"  - False Negatives (FN): {fn} (Incorrectly predicted 'not_center')")
    print(f"  - True Positives (TP):  {tp} (Correctly predicted 'center')")
    print("---------------------------------")


def generate_full_report(model, test_data_dir, label_map, output_csv_path):
    """
    This function performs all analysis tasks and returns labels for the summary.
    """
    print("\nStarting full analysis...")
    mismatch_count = 0
    all_true_labels = []
    all_predicted_labels = []
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['Image Path', 'Image Name', 'True Class', 'Predicted Class', 'Confidence', 'Correct/Incorrect', 'Log Loss']
            csv_writer.writerow(header)

            for folder_name, true_class in label_map.items():
                folder_path = os.path.join(test_data_dir, folder_name)
                if not os.path.isdir(folder_path):
                    print(f"Warning: Directory not found, skipping: {folder_path}")
                    continue

                print(f"--- Processing folder: '{folder_name}' ---")
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    
                    predicted_class, confidence, _ = get_prediction_details(model, image_path)
                    
                    if predicted_class is None: continue
                    
                    all_true_labels.append(true_class)
                    all_predicted_labels.append(predicted_class)
                    
                    if predicted_class != true_class:
                        mismatch_count += 1
                        print(f"🚨 Mismatch #{mismatch_count}: '{image_name}' was predicted as {predicted_class}")
                    
                    is_correct_flag = 'Correct' if predicted_class == true_class else 'Incorrect'
                    log_loss = calculate_log_loss(true_class, confidence)
                    
                    csv_writer.writerow([image_path, image_name, true_class, predicted_class, f'{confidence:.4f}', is_correct_flag, f'{log_loss:.4f}'])
        
        print("\n--- Analysis Complete ---")
        if mismatch_count == 0:
            print("✅ No mismatches found during processing.")
        else:
            print(f"Found {mismatch_count} potential mismatches.")
        print(f"✅ Successfully created report at: {output_csv_path}")
        
        return all_true_labels, all_predicted_labels

    except Exception as e:
        print(f"❌ Failed during analysis. Error: {e}")
        return [], []