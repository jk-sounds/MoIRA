import argparse
import json
import os
import sys
from transformers import AutoTokenizer

# Ensure we can find the Omni-Mol metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Omni-Mol'))

try:
    from metrics import calculate_text_scores
except ImportError:
    print("Error: Could not import metrics from Omni-Mol. Please check the path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the prediction jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the metrics json")
    parser.add_argument("--tokenizer_path", type=str, default="/home/public_space/zhangxiaohong/yintaoo/vicuna-7b", help="Path to tokenizer")
    args = parser.parse_args()

    # Read predictions
    data = []
    # Generate sample outputs a JSON list (despite .jsonl extension in some scripts, generate_sample.py uses json.dump)
    # But let's handle both just in case.
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
        # Fallback to JSONL reading
        with open(args.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append({
                        "gt": str(item.get('gt_self', '')), 
                        "pred": str(item.get('pred_self', '')),
                        "prompt": item.get('prompt', '')
                    })

    print(f"Calculating text metrics for {len(data)} samples...")
    
    # Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Calculate Text Metrics (BLEU, METEOR, ROUGE)
    metrics = calculate_text_scores(data, save_path=None, tokenizer=tokenizer)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
