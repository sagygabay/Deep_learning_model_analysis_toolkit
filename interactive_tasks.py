# interactive_fixer.py (with clickable image paths and file logging)
# An interactive tool to review misclassified images and correct labels.

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def _get_current_metrics(true_labels, pred_labels):
    """Calculates and returns the current metrics as a tuple."""
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return accuracy, f1

def _print_live_metrics(accuracy, f1):
    """A helper function to print updated metrics in a standard format."""
    print("---------------------------------")
    print(f"🚀 New Live Metrics:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    print("---------------------------------\n")

def run_interactive_session():
    """Main function to start the interactive label correction session."""
    
    # --- Configuration ---
    CSV_REPORT_PATH = 'prediction_report.csv'
    LOG_FILE_PATH = 'change_log.txt'
    # ---------------------

    print("--- Interactive Label Fixer ---")
    
    # Step 1: Load the report
    if not os.path.exists(CSV_REPORT_PATH):
        print(f"❌ Error: Report file not found at '{CSV_REPORT_PATH}'.")
        print("Please run the main analysis script first to generate the report.")
        return

    df = pd.read_csv(CSV_REPORT_PATH)
    print(f"Loaded report with {len(df)} entries.")
    
    # Step 2: Find and sort misclassified images
    incorrect_df = df[df['Correct/Incorrect'] == 'Incorrect'].copy()
    if incorrect_df.empty:
        print("✅ No misclassified images found in the report. Excellent!")
        return

    incorrect_df['confidence_error'] = incorrect_df.apply(
        lambda row: row['Confidence'] if row['True Class'] == 0 else 1 - row['Confidence'], 
        axis=1
    )
    sorted_incorrect = incorrect_df.sort_values(by='confidence_error', ascending=False)
    
    print(f"\nFound {len(sorted_incorrect)} misclassified images to review.")
    
    # Step 3: Prepare live lists and the change log
    live_true_labels = df['True Class'].tolist()
    pred_labels = df['Predicted Class'].tolist()
    change_log = [] 
    
    # Step 4: Loop through the images for review
    for i, (original_df_index, row) in enumerate(sorted_incorrect.iterrows()):
        print("\n=======================================================")
        print(f"Reviewing Image #{i + 1}/{len(sorted_incorrect)}")
        
        # --- THIS IS THE NEW LINE ---
        # Print the clickable path to the image being reviewed.
        print(f"  - View Image:      {os.path.abspath(row['Image Path'])}")
        # ----------------------------
        
        print(f"  - File:            {row['Image Name']}")
        print(f"  - Original Label:    {row['True Class']}")
        print(f"  - Model Predicted:   {row['Predicted Class']} (Confidence: {row['Confidence']:.4f})")
        
        choice = input("  - Action -> (c)hange label, (k)eep current label, (q)uit review: ").lower().strip()
        
        if choice == 'q':
            print("Quitting review session.")
            break
        elif choice == 'k':
            print("  -> Label kept as is.")
            continue
        elif choice == 'c':
            original_label = live_true_labels[original_df_index]
            new_label = 1 - original_label
            live_true_labels[original_df_index] = new_label
            
            new_accuracy, new_f1 = _get_current_metrics(live_true_labels, pred_labels)
            
            change_log.append({
                'image_name': row['Image Name'],
                'from_label': original_label,
                'to_label': new_label,
                'accuracy_after': new_accuracy,
                'f1_after': new_f1
            })
            
            print(f"  -> SUCCESS: Changed label for '{row['Image Name']}' from {original_label} to {new_label}")
            _print_live_metrics(new_accuracy, new_f1)
        else:
            print("  -> Invalid choice. Keeping label as is.")
            
    # Step 5: Final summary and saving files
    print("\n=======================================================")
    print("Review session finished.")
    
    if change_log:
        print("\n--- Summary of Changes Made ---")
        
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("--- Log of Label Corrections ---\n\n")
            for i, change in enumerate(change_log):
                print(f"Change #{i+1}: Image '{change['image_name']}' label flipped from {change['from_label']} to {change['to_label']}")
                log_entry = (
                    f"Change #{i+1}:\n"
                    f"  Image:      {change['image_name']}\n"
                    f"  Label Flip: {change['from_label']} -> {change['to_label']}\n"
                    f"  Resulting F1: {change['f1_after']:.4f}, Accuracy: {change['accuracy_after']:.4f}\n"
                    "------------------------------------\n\n"
                )
                f.write(log_entry)
        
        full_log_path = os.path.abspath(LOG_FILE_PATH)
        print("\n✅ A detailed text log of these changes has been saved.")
        print(f"   Log File Path: {full_log_path}")
        
        print("-" * 35)
        save_choice = input("Do you also want to save the new, fully corrected CSV report? (y/n): ").lower().strip()
        if save_choice == 'y':
            df['Corrected True Class'] = live_true_labels
            df['Corrected Correct/Incorrect'] = df.apply(
                lambda r: 'Correct' if r['Predicted Class'] == r['Corrected True Class'] else 'Incorrect',
                axis=1
            )
            corrected_csv_path = 'prediction_report_v2_corrected.csv'
            df.to_csv(corrected_csv_path, index=False)
            print(f"✅ Successfully saved corrected CSV report to '{corrected_csv_path}'")
    else:
        print("No changes were made.")

if __name__ == "__main__":
    run_interactive_session()