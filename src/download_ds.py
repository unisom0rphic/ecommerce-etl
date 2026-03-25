import os
import shutil

import kagglehub

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("carrie1/ecommerce-data")

    print("Path to dataset files:", path)

    filename = os.listdir(path)[0]

    print(f"Filename: {filename}")

    full_path = f"{path}/{filename}"

    print(f"Full path to file: {full_path}")

    try:
        shutil.move(full_path, "./data")
        print("Success!")
    except Exception as e:
        print(f"ERROR! {e}")
