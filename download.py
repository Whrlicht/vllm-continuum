from datasets import load_dataset

ds = load_dataset(
    "allenai/WildChat-1M",
    cache_dir="/data/whr/vllm-continuum/trace_data/origin_data/general/wildchat",
)