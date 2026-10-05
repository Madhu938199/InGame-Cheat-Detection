"""
Master Setup & Pipeline Orchestrator
Project: In-Game Cheat Detection System (Aegis Anti-Cheat)
File: setup_project.py

Executes complete setup on the real dataset/final_dataset.csv:
1. Verifies dataset presence and schema integrity.
2. Runs Stage 1 & Stage 2 Exploratory Data Analysis & Generates Plots.
3. Fits the unified Stage 3 Preprocessing Pipeline (ml/model/preprocessing_pipeline.pkl).
4. Trains and benchmarks the 4 models (Stage 4 & 5), selecting and saving the best classifier (ml/model/cheat_detection_model.pkl).
5. Exports risk thresholds and model metadata (ml/model/metadata.json).
6. Initializes database schemas and demonstration seed accounts.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def run_master_setup():
    print("=" * 80)
    print(" IN-GAME CHEAT DETECTION SYSTEM - MASTER PIPELINE SETUP")
    print("=" * 80)

    dataset_path = os.path.join(BASE_DIR, "dataset", "final_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"Error: Real dataset not found at {dataset_path}!")
        print("Please ensure final_dataset.csv is in the dataset/ directory.")
        return

    # 1. Run Stages 4, 5, 6: Training & Evaluation on the Real Dataset
    print("\n[Step 1/3] Running Feature Selection, Group-Aware Split, Model Benchmarking...")
    from ml.train_and_evaluate import execute_training_and_evaluation
    results = execute_training_and_evaluation(dataset_path=dataset_path)

    # 2. Run Database Setup
    print("\n[Step 2/3] Initializing Database & Demonstration Accounts...")
    from backend.database import init_db
    init_db()

    # 3. Completion Summary
    print("\n" + "=" * 80)
    print(" MASTER SETUP COMPLETE!")
    print("=" * 80)
    print(f"Selected Production Model: {results['best_model_name']}")
    print(f"Artifacts Saved in:        ml/model/")
    print(f"  • cheat_detection_model.pkl")
    print(f"  • preprocessing_pipeline.pkl")
    print(f"  • metadata.json")
    print("\nTo launch the Flask Web Dashboard & REST API:")
    print("   py backend/app.py   (or python backend/app.py)")
    print("\nTo run the automated test suite:")
    print("   py tests/test_system.py   (or python tests/test_system.py)")
    print("\nDefault Presentation Accounts:")
    print("   • Player:    player1    / password123")
    print("   • Moderator: moderator1 / password123")
    print("=" * 80)

if __name__ == '__main__':
    run_master_setup()
