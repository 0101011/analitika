import os
import sys
from config import get_config
import analitika

def run_demo():
    print("Running demo with default configuration...")
    analitika.process_data()

    print("\nRunning demo with custom tokenizer...")
    config = get_config()
    config['TOKENIZER'] = 'custom'
    analitika.CONFIG = config
    analitika.process_data()

    print("\nRunning demo with data augmentation disabled...")
    config = get_config()
    config['ENABLE_AUGMENTATION'] = False
    analitika.CONFIG = config
    analitika.process_data()

if __name__ == '__main__':
    run_demo()