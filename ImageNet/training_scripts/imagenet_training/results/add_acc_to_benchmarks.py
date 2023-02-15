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
    default="./benchmark_inference_GTX1080_fp32_small_torch1.10.csv",
    type=str,
    metavar="FILENAME",
    help="the csv file for which you want to add accuracy",
)


def add_acc_to_csv(imagenet_results, csv_filename):
    df_imagenet_results = pd.read_csv(imagenet_results)
    df_imagenet_accs = df_imagenet_results[["model", "top1", "top5"]]
    df_csv = pd.read_csv(csv_filename)
    df_csv_acc = pd.merge(df_csv, df_imagenet_accs, on=["model"])
    df_csv_acc.to_csv(csv_filename.replace(".csv", "_with_accuracy.csv"), index=False)
    print(f"{csv_filename} is done")


if __name__ == "__main__":
    args = parser.parse_args()
    add_acc_to_csv(args.imagenet_results, args.bench_csv)
