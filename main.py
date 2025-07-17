# main.py (Final Version with Interactive Menu)

import sys

# Import our modular functions from all task-specific files
from load_model import LOAD_MODEL
from analysis_tasks import generate_full_report, print_summary_metrics
from visualization_tasks import create_all_visuals
from interactive_tasks import run_interactive_session

def display_menu():
    """Prints the main user menu to the console."""
    print("\n=====================================")
    print("  Model Analysis Interactive Menu")
    print("=====================================")
    print("1. Run Full Pipeline (Analyze + Visualize)")
    print("2. Run Analysis Only (Generate CSV Report)")
    print("3. Run Visualization Only (From existing CSV)")
    print("4. Run Interactive Label Fixer")
    print("5. Exit")
    print("-------------------------------------")

def main():
    """
    Main execution function with an interactive menu to control tasks.
    """
    # --- Master configuration ---
    MODEL_PATH =r'C:\Users\avimo\Desktop\Sagy-Deep\anaylist_tools\deep_learning_model_test_analysis\model_b3\best_model.pth'
    TEST_DATA_DIR =r'C:/Users/avimo/Desktop/Sagy-Deep/models_train/test'
    OUTPUT_CSV_PATH = 'prediction_report.csv'
    OUTPUT_VIS_DIR = 'output_visualizations'
    CHANGE_LOG_PATH = 'change_log.txt'
    LABEL_MAP = {'center': 1, 'not_center': 0}
    # --------------------------

    # Loop indefinitely until the user chooses to exit
    while True:
        display_menu()
        choice = input("Please enter your choice (1-5): ")

        if choice == '1':
            # --- Run Full Pipeline ---
            print("\n--- Running Full Analysis and Visualization Pipeline ---")
            my_model = LOAD_MODEL(MODEL_PATH)
            if my_model:
                true_labels, pred_labels = generate_full_report(model=my_model, test_data_dir=TEST_DATA_DIR, label_map=LABEL_MAP, output_csv_path=OUTPUT_CSV_PATH)
                if true_labels and pred_labels:
                    print_summary_metrics(true_labels, pred_labels)
                create_all_visuals(csv_path=OUTPUT_CSV_PATH, output_dir=OUTPUT_VIS_DIR)
            else:
                print("Halting pipeline: Model could not be loaded.")
            input("\nPress Enter to return to the main menu...")

        elif choice == '2':
            # --- Run Analysis Only ---
            print("\n--- Running Analysis Task Only ---")
            my_model = LOAD_MODEL(MODEL_PATH)
            if my_model:
                true_labels, pred_labels = generate_full_report(model=my_model, test_data_dir=TEST_DATA_DIR, label_map=LABEL_MAP, output_csv_path=OUTPUT_CSV_PATH)
                if true_labels and pred_labels:
                    print_summary_metrics(true_labels, pred_labels)
            else:
                print("Halting task: Model could not be loaded.")
            input("\nPress Enter to return to the main menu...")

        elif choice == '3':
            # --- Run Visualization Only ---
            print("\n--- Running Visualization Task Only ---")
            create_all_visuals(csv_path=OUTPUT_CSV_PATH, output_dir=OUTPUT_VIS_DIR)
            input("\nPress Enter to return to the main menu...")

        elif choice == '4':
            # --- Run Interactive Fixer ---
            print("\n--- Running Interactive Fixer Task ---")
            run_interactive_session(csv_path=OUTPUT_CSV_PATH, log_path=CHANGE_LOG_PATH)
            input("\nPress Enter to return to the main menu...")

        elif choice == '5':
            # --- Exit ---
            print("Exiting the application. Goodbye!")
            sys.exit()

        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()