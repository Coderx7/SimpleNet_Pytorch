import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="A small utility to merge model accuracy with timm benchmarks")
parser.add_argument(
    "--imagenet-results",
    default="./results-imagenet.csv",
    type=str,
    metavar="FILENAME",
    help="the imagenet results csv file to get the accuracies from",
)
parser.add_argument(
    "--bench-csv",
    default="",
    type=str,
    metavar="FILENAME",
    help="the csv file for which you want to add accuracy",
)


def add_acc_to_csv(imagenet_results, csv_filename):
    df_imagenet_results = pd.read_csv(imagenet_results)
    df_imagenet_accs = df_imagenet_results[["model", "top1", "top5"]]
    df_csv = pd.read_csv(csv_filename)
    df_csv_acc = pd.merge(df_csv, df_imagenet_accs, on=["model"])
    df_csv_acc.to_csv(csv_filename.replace(".csv", "_acc.csv"), index=False)
    print(f"--{csv_filename:<60} is processed.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.bench_csv:
        add_acc_to_csv(args.imagenet_results, args.bench_csv)
    else:
        print('Fetching all benchmark logs and adding accuracy to all in bulk...')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for file in os.listdir(current_dir):
            if 'bench' in file and file.endswith('.csv'):
                add_acc_to_csv(args.imagenet_results, file)
        print(f'all done.')
