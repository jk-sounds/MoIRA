import argparse
import json
import os
import sys

# Ensure we can find the Omni-Mol metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Omni-Mol'))

try:
    from metrics import calculate_mae_with_text
except ImportError:
    print("Error: Could not import metrics from Omni-Mol. Please check the path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the prediction jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the metrics json")
    args = parser.parse_args()

    # Read predictions
    data = []
    try:
        with open(args.input_file, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                 data.append({
                    "gt": str(item.get('gt_self', '')), 
                    "pred": str(item.get('pred_self', '')),
                    "prompt": item.get('prompt', '')
                })
    except json.JSONDecodeError:
        with open(args.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append({
                        "gt": str(item.get('gt_self', '')), 
                        "pred": str(item.get('pred_self', '')),
                        "prompt": item.get('prompt', '')
                    })

    print(f"Calculating metrics for {len(data)} samples...")

    # Calculate MAE
    metrics = calculate_mae_with_text(data, save_path=None)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
