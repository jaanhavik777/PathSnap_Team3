"""Train/evaluate the Person 4 MLP on train and student_test only."""
import argparse
import json
from pathlib import Path
import shutil
from src.models.train_mlp import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",required=True,help="Dataset root containing train/ and student_test/")
    parser.add_argument("--output",default="artifacts/models/mlp")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--candidate-count",type=int,default=4)
    args=parser.parse_args()
    out=Path(args.output)
    result=run(args.data_root,out,args.seed,args.candidate_count)
    metrics=Path("artifacts/metrics"); metrics.mkdir(parents=True,exist_ok=True)
    plots=Path("artifacts/plots"); plots.mkdir(parents=True,exist_ok=True)
    for name in ("metrics.json","metrics.csv"):
        if (out/name).exists(): shutil.copy2(out/name,metrics/f"mlp_{name}")
    for name in ("training_loss.png","confusion_matrix.png","performance.png"):
        if (out/name).exists(): shutil.copy2(out/name,plots/f"mlp_{name}")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
