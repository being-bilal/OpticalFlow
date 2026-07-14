from huggingface_hub import hf_hub_download

repo_id = "ntnu-arl/underwater-datasets"

bag_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="subset-mclab/mclab_1/mclab_1.bag"
)

gt_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="subset-mclab/mclab_1/mclab_1_baseline.tum"
)

print(f"Bag downloaded to: {bag_path}")
print(f"Ground truth downloaded to: {gt_path}")