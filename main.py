import argparse
import os
from src.DrowsinessDetector import DrowsinessDetector


def main():
    
    #root_directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # parse different arguments
    parser = argparse.ArgumentParser(description='Drowsiness Detection System')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default=os.path.join(root_dir, 'models','best_model.keras'),
        help='Path to the trained model file (default: ../models/best_model.keras)'
    )
    
    parser.add_argument(
        '--alarm', 
        type=str, 
        default=os.path.join(root_dir, 'Alert.wav'),
        help='Path to the alarm sound file (default: ../Alert.wav)'
    )
    
    parser.add_argument(
        '--drowsy_time', 
        type=float, 
        default=0.2,
        help='Time threshold (in seconds) to detect drowsiness (default: 0.2)'
    )
    args = parser.parse_args()
    
    try:
        # Initializing detector with user inputs
        detector = DrowsinessDetector(
            model_path=args.model,
            alarm_path=args.alarm,
            drowsy_time=args.drowsy_time
        )
        
        print("Detecting Drowsiness....")
        print("Please press 'q' to stop detection")
        
        # detecting drowsiness
        detector.detect_drowsiness()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()