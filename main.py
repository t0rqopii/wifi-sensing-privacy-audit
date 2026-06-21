import argparse
import sys
from src.core.csi_generator import CSIGenerator
from src.utils.data_loader import DataLoader
from src.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="CSI Privacy Framework")
    parser.add_argument('--mode', type=str, choices=['generate', 'analyze', 'mitigate', 'evaluate', 'full'],
                        default='full', help='Pipeline mode to run')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples per activity class')
    
    args = parser.parse_args()
    
    if args.mode == 'full' or args.mode == 'evaluate':
        evaluator = Evaluator()
        evaluator.run_full_evaluation(samples_per_class=args.samples)
    elif args.mode == 'generate':
        print(f"Generating synthetic dataset with {args.samples} samples per class...")
        generator = CSIGenerator()
        X, y, classes = generator.generate_dataset(samples_per_class=args.samples)
        loader = DataLoader()
        loader.save_dataset("synthetic_csi_data.npz", X, y, classes)
    elif args.mode in ['analyze', 'mitigate']:
        print(f"Mode '{args.mode}' is best run via the 'full' evaluation pipeline.")
        print("Running full pipeline instead...")
        evaluator = Evaluator()
        evaluator.run_full_evaluation(samples_per_class=args.samples)
        
if __name__ == "__main__":
    main()
