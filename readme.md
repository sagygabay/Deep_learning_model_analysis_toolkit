# Deep Learning Model Analysis Toolkit

A complete, interactive command-line application designed to analyze, visualize, and correct the predictions of a binary classification model built with PyTorch and timm.
This toolkit provides a seamless workflow from initial performance assessment to hands-on data cleaning, helping you understand your model's failures and improve your dataset's quality.

## Features
✅ Comprehensive Analysis: Processes a test dataset and generates a detailed CSV report containing image paths, true vs. predicted labels, confidence scores, and per-image log loss.
✅ Performance Metrics: Automatically calculates and displays a final summary including F1 Score, Accuracy, Precision, Recall, and a full Confusion Matrix.
✅ Data Visualization: Generates a set of publication-ready plots from the analysis report, including:
Confusion Matrix Heatmap
Confidence Distribution Histogram
ROC Curve with AUC Score
✅ Interactive Label Correction: A powerful interactive session that finds all misclassified images, sorts them by the model's error confidence, and allows you to:
View a clickable path to each incorrect image.
Interactively correct the label.
See the F1 Score and Accuracy update in real-time with every change.
✅ Persistent Logging: All label corrections made during an interactive session are saved to a change_log.txt file, providing a complete history of your work.
✅ Modular & Clean Code: The project is broken down into single-responsibility Python modules, making it easy to understand, maintain, and extend.
✅ User-Friendly Interface: The entire tool is controlled from a single main.py script with a simple, interactive menu.
## Project Structure
Your project folder should be organized as follows for the tool to work correctly:
your_project_folder/
├── model_b3/best_model.pth         # Your trained model checkpoint file
├── main.py                         # The single, interactive entry point to the application
├── load_model.py                   # Module for loading the PyTorch model
├── prediction_utils.py             # Module for running a prediction on a single image
├── analysis_tasks.py               # Module for generating the CSV report and metrics
├── visualization_tasks.py          # Module for generating plots from the report
├── interactive_tasks.py            # Module for the interactive label fixer session
├── output_visualizations/          # (Auto-created) Folder for saved plots
│   ├── confusion_matrix.png
│   └── ...
├── prediction_report.csv           # (Auto-created) The main analysis report
├── prediction_report_corrected.csv # (Optional) Saved after an interactive session
└── change_log.txt                  # (Optional) Saved after an interactive session
## Setup and Installation
Follow these steps to get the toolkit up and running.
1. Get the Code:
Place all the Python script files (main.py, load_model.py, etc.) into your project directory.
2. Set up the Python Environment:
This tool was developed using an Anaconda environment. It is recommended to use a virtual environment to keep dependencies clean.
Install all required libraries using pip:
```bash
pip install torch timm pandas scikit-learn matplotlib seaborn
```
3. Place Your Model:
Make sure your trained model file (e.g., model_b3/best_model.pth) is placed in the project directory or that the path to it is correct.
4. Configure Paths in main.py:
Open the main.py script and review the Master Configuration section at the top. Ensure the following paths are correct for your system:
MODEL_PATH: The path to your trained model file.
TEST_DATA_DIR: The path to your test dataset folder (which should contain center and not_center subfolders).
Usage
The entire toolkit is operated from the main.py script, which provides a user-friendly menu.
To start the application, run:
```bash
python main.py
```

You will be presented with the following menu:
```
=====================================
  Model Analysis Interactive Menu
=====================================
1. Run Full Pipeline (Analyze + Visualize)
2. Run Analysis Only (Generate CSV Report)
3. Run Visualization Only (From existing CSV)
4. Run Interactive Label Fixer
5. Exit
-------------------------------------
Please enter your choice (1-5):
```
- **Option 1:** The default workflow. It first runs the analysis to create the CSV report and then immediately generates all visualizations from that report.
- **Option 2:** Use this if you only need the raw data. It will process all images and create the `prediction_report.csv` file, then print the final F1/Accuracy summary.
- **Option 3:** Use this if you already have a report and just want to re-generate the plots without re-processing all the images.
- **Option 4:** The data-cleaning tool. This starts the powerful interactive session to review and correct mislabeled images.
- **Option 5:** Exits the application.

## Example Workflow

A typical use case for this toolkit would be:

1.  **Run Initial Analysis:** Run the script and choose **Option 2** to get a baseline understanding of your model's performance and generate the first `prediction_report.csv`.
2.  **Get a Visual Overview:** Run the script again and choose **Option 3**. Check the generated plots, especially the `confusion_matrix.png`, to see where the model is failing.
3.  **Correct Mislabels:** After reviewing the plots, run the script and choose **Option 4**. The tool will guide you through the model's biggest mistakes. You can **Ctrl+Click** the image paths to view them and decide if the ground truth label is wrong.
4.  **Save Corrections:** After the interactive session, save the new `_corrected.csv` report and the `change_log.txt`.
5.  **Re-evaluate (Optional):** You can now use the corrected CSV to see how your "true" metrics have changed, or even use the improved labels to retrain a better model.

## Modules Overview

- `main.py`: The orchestrator and user interface. It takes user input and calls functions from other modules.
- `load_model.py`: Handles the specific, complex logic of loading your saved model checkpoint and remapping its layers.
- `prediction_utils.py`: Contains the core, single-responsibility function for getting a prediction from the model for one image.
- `analysis_tasks.py`: Contains the high-level logic for processing the entire dataset, generating the CSV, and calculating summary metrics like F1 score.
- `visualization_tasks.py`: Reads the final CSV report and generates all plots using `matplotlib` and `seaborn`.
- `interactive_tasks.py`: Contains the logic for the interactive data cleaning session, including displaying prompts and recalculating metrics on the fly.

## Future Improvements

- **Image Display:** Use a library like `Pillow` or `OpenCV` to automatically open and display the image being reviewed in the interactive fixer, instead of just printing a path.
- **File System Operations:** Add an option in the interactive fixer to automatically move a file with a corrected label to a different folder (e.g., from `/test/center` to `/test/not_center`).
- **Multi-Class Support:** Adapt the logic to handle models with more than two output classes. This would primarily involve changing from `sigmoid` to `softmax` and updating the metrics calculations.
