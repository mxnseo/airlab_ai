import os
import random
import shutil

# 데이터셋 경로
images_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\origin\images"
labels_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\origin\labels"

# 데이터셋을 저장할 폴더 경로
dataset1_images_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\dataset1\images"
dataset1_labels_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\dataset1\labels"
dataset2_images_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\dataset2\images"
dataset2_labels_dir = r"D:\MINSEO\ultralytics\datasets\TLD_2024\dataset2\labels"

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

# 파일 일치 여부 확인
assert len(image_files) == len(label_files), "이미지와 라벨 파일 개수가 일치하지 않습니다."
for img_file, lbl_file in zip(image_files, label_files):
    assert os.path.splitext(img_file)[0] == os.path.splitext(lbl_file)[0], f"{img_file}와 {lbl_file}의 이름이 일치하지 않습니다."

combined = list(zip(image_files, label_files))
random.shuffle(combined)

total_size = len(combined)
dataset_size = total_size // 2

dataset1_data = combined[:dataset_size]
dataset2_data = combined[dataset_size:]

# dataset1
os.makedirs(dataset1_images_dir, exist_ok=True)
os.makedirs(dataset1_labels_dir, exist_ok=True)
for img_file, lbl_file in dataset1_data:
    shutil.copy(os.path.join(images_dir, img_file), os.path.join(dataset1_images_dir, img_file))
    shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(dataset1_labels_dir, lbl_file))
    print(f"Dataset 1: {img_file} -> {dataset1_images_dir}, {lbl_file} -> {dataset1_labels_dir}")

# dataset2
os.makedirs(dataset2_images_dir, exist_ok=True)
os.makedirs(dataset2_labels_dir, exist_ok=True)
for img_file, lbl_file in dataset2_data:
    shutil.copy(os.path.join(images_dir, img_file), os.path.join(dataset2_images_dir, img_file))
    shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(dataset2_labels_dir, lbl_file))
    print(f"Dataset 2: {img_file} -> {dataset2_images_dir}, {lbl_file} -> {dataset2_labels_dir}")
