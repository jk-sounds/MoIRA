import argparse
import json
import os
import sys
from tqdm import tqdm

# Add the parent directory to sys.path to allow importing from Omni-Mol
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Omni-Mol'))

from metrics import calculate_reaction_metrics

def main(args):
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Adapt data format if necessary, but generate_sample.py output should match what metrics.py expects
    # metrics.py expects list of dicts with 'pred' and 'gt' (or 'gt_self'/'pred_self' keys might need mapping)
    
    # generate_sample.py outputs: {"prompt": ..., "gt_self": ..., "pred_self": ...}
    # metrics.py's calculate_reaction_metrics expects: {"gt": ..., "pred": ...}
    
    formatted_data = []
    for item in data:
        formatted_data.append({
            "gt": item.get("gt_self", item.get("gt")),
            "pred": item.get("pred_self", item.get("pred"))
        })

    metrics = calculate_reaction_metrics(
        formatted_data, 
        save_path=args.output_file, 
        morgan_r=2, 
        eos_token=None # EOS handling might be done in generation or here. 
                       # generate_sample.py usually strips special tokens.
    )
    
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the inference output jsonl/json file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the metrics json file")
    args = parser.parse_args()
    main(args)
