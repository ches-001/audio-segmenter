import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
import kaggle

if __name__ == "__main__":
    dataset_url = "https://www.kaggle.com/datasets/dogrose/musan-dataset"
    dataset_dir = "data"
    os.makedirs(dataset_dir,  exist_ok=True)
    kaggle.api.dataset_download_cli(dataset_url.split("datasets/")[-1], path=dataset_dir, unzip=True)