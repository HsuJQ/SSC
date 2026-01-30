"""
Pangu Goal Interpretation (Modular Entrypoint)
"""
import logging
from ssc.pangu.goal_interpretation.constants import LOG_FILE
from ssc.pangu.goal_interpretation.runner import process_all_tasks

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    process_all_tasks()

if __name__ == "__main__":
    main()
