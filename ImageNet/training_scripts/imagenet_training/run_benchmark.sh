echo 'benchmarking list_small.txt'
python benchmark.py --model-list model_list_small.txt --bench inference --channels-last --results-file results/benchmark_inference_RTX3080_fp32_small.csv

echo 'benchmarking list_normal_known'
python benchmark.py --model-list model_list_normal_known.txt --bench inference --channels-last --results-file results/benchmark_inference_RTX3080_fp32_normal_known.csv

echo 'benchmarking list_normal'
python benchmark.py --model-list model_list_normal.txt --bench inference --channels-last --results-file results/benchmark_inference_RTX3080_fp32_normal.csv

echo 'all done'
