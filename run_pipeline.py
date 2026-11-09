"""
One-command End-to-End Orchestrator for LoanWatch NBFC Early Warning System.
Executes: Data Generation -> Feature Engineering -> Model Training & Calibration ->
Tableau Exports -> Test Suite -> Optional Web App Launch.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LoanWatch-Pipeline")


def run_step(step_name: str, cmd_args: list) -> None:
    """Executes a pipeline step as a subprocess, ensuring fail-fast behavior."""
    logger.info("==================================================")
    logger.info("▶ STEP: %s", step_name)
    logger.info("Command: %s", " ".join(cmd_args))
    logger.info("==================================================")

    res = subprocess.run(cmd_args, check=False)
    if res.returncode != 0:
        logger.error("❌ Step '%s' failed with exit code %d. Aborting pipeline.", step_name, res.returncode)
        sys.exit(res.returncode)
    logger.info("✅ Step '%s' completed successfully.\n", step_name)


def main():
    parser = argparse.ArgumentParser(description="LoanWatch End-to-End Execution Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip synthetic data generation if already created")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training if artifacts already exist")
    parser.add_argument("--launch", action="store_true", help="Launch Streamlit web application upon completion")
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. Synthetic Data Generation
    if not args.skip_data:
        run_step("1. Synthetic Data Generation", [python_exe, "-m", "src.data_generation"])
    else:
        logger.info("Skipping data generation as requested.")

    # 2. Point-in-Time Feature Engineering
    run_step("2. Point-in-Time Feature Engineering", [python_exe, "-m", "src.features"])

    # 3. Model Training, Calibration & Evaluation
    if not args.skip_train:
        run_step("3. Model Training, Calibration & Evaluation", [python_exe, "-m", "src.train"])
    else:
        logger.info("Skipping model training as requested.")

    # 4. Generate Tableau Exports
    run_step("4. Pre-Aggregated Tableau Exports", [python_exe, "-m", "src.exports"])

    # 5. Run Automated Test Suite
    run_step("5. Automated Test Suite (pytest)", [python_exe, "-m", "pytest", "-v"])

    logger.info("🎉 All LoanWatch pipeline steps succeeded!")

    # 6. Optional Streamlit Web Application Launch
    if args.launch:
        logger.info("Launching Streamlit Application on http://localhost:8501 ...")
        subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
