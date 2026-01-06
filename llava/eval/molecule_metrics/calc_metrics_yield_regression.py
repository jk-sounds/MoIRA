import argparse
import json
import os
import sys

try:
    from metrics import calculate_mae_with_text, calculate_r2
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
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # Mapping generic keys to what metrics.py expects
                    # metrics.py expects: 'gt' and 'pred' keys in list[dict]
                    # generate_sample.py outputs: 'gt_self' and 'pred_self'
                    
                    data.append({
                        "gt": str(item.get('gt_self', '')), # Ensure string format for extraction
                        "pred": str(item.get('pred_self', '')),
                        "prompt": item.get('prompt', '')
                    })
                except json.JSONDecodeError:
                    continue
                    
    # Flatten if it was a list of lists (which shouldn't happen with jsonl line reading, 
    # but generate_sample output format might vary. Wait, generate_sample outputs a JSON LIST, not JSONL!)
    # Let's re-check generate_sample.py
    # json.dump(outs, ans_file, indent=2) -> It writes a JSON list, not JSONL.
    
    # Re-reading as JSON list
    with open(args.input_file, 'r') as f:
        raw_data = json.load(f)
        
    data = []
    for item in raw_data:
        data.append({
            "gt": str(item.get('gt_self', '')),
            "pred": str(item.get('pred_self', '')),
            "prompt": item.get('prompt', '')
        })

    print(f"Calculating metrics for {len(data)} samples...")

    # Calculate MAE
    mae_metrics = calculate_mae_with_text(data, save_path=None)
    
    # Calculate R2
    r2_metrics = calculate_r2(data, save_path=None)
    
    # Merge metrics
    combined_metrics = {**mae_metrics, **r2_metrics}
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
        
    print("Metrics:")
    print(json.dumps(combined_metrics, indent=2))
    print(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
