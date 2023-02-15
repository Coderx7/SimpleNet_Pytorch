echo 'benchmarking list_small.txt'
python benchmark.py --model-list model_list_small.txt --bench inference --channels-last --results-file results/benchmark_inference_GTX1080_fp32_small_torch1.11.csv

echo 'benchmarking list_normal_known'
python benchmark.py --model-list model_list_normal_known.txt --bench inference --channels-last --results-file results/benchmark_inference_GTX1080_fp32_normal_known_torch1.11.csv

echo 'benchmarking list_normal'
python benchmark.py --model-list model_list_normal.txt --bench inference --channels-last --results-file results/benchmark_inference_GTX1080_fp32_normal_torch1.11.csv

echo 'all done'
