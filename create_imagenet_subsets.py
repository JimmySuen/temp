import os
import shutil
from pathlib import Path

src_folder = "/mnt/blobfuse/data/ImageNet/"

txt_file = "/mnt/blobfuse/projects/simclr/imagenet_subsets/1percent.txt"


dst_folder = "/mnt/blobfuse/data/ImageNetS1/"


with open(txt_file, 'r') as f:
    Lines = f.readlines()
    count = 0
    for line in Lines:
        count += 1
        name = line.strip()
        folder = name.split("_")[0]
        print("Line{}: {}".format(count, name))
        src = os.path.join(src_folder, 'train', folder, name)
        dst = os.path.join(dst_folder, 'train', folder)
        Path(dst).mkdir(parents=True, exist_ok=True)
        dst_file = os.path.join(dst, name)
        shutil.copyfile(src, dst_file)


txt_file = "/mnt/blobfuse/projects/simclr/imagenet_subsets/10percent.txt"
dst_folder = "/mnt/blobfuse/data/ImageNetS10/"

with open(txt_file, 'r') as f:
    Lines = f.readlines()
    count = 0
    for line in Lines:
        count += 1
        name = line.strip()
        folder = name.split("_")[0]
        print("Line{}: {}".format(count, name))
        src = os.path.join(src_folder, 'train', folder, name)
        dst = os.path.join(dst_folder, 'train', folder)
        Path(dst).mkdir(parents=True, exist_ok=True)
        dst_file = os.path.join(dst, name)
        shutil.copyfile(src, dst_file)

