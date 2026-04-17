from datasets import load_dataset

ds = load_dataset(
    "SWE-bench/SWE-smith-trajectories",
    cache_dir="/data/whr/datasets/SWE-bench/SWE-smith-trajectories",
)