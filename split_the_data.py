import os
import shutil
import random
 
#데이터셋 경로 입력.
images_train_dir = 'D:\\ChanWooKim\\dataset\\Compete_COCO\\images\\train'
labels_train_dir = 'D:\\ChanWooKim\\dataset\\Compete_COCO\\labels\\train'
val_images_dir = 'D:\\ChanWooKim\\dataset\\Compete_COCO\\images\\val'
val_labels_dir = 'D:\\ChanWooKim\\dataset\\Compete_COCO\\labels\\val'
 
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
 
image_files = os.listdir(images_train_dir)
 
random.shuffle(image_files)
 
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)
 
train_images = image_files[:split_index]
val_images = image_files[split_index:]
 
for val_image in val_images:
    shutil.move(os.path.join(images_train_dir, val_image), os.path.join(val_images_dir, val_image))
    
    label_file = val_image.replace('.png', '.txt') 
    shutil.move(os.path.join(labels_train_dir, label_file), os.path.join(val_labels_dir, label_file))
 
print(f"훈련 데이터: {len(train_images)}개, 검증 데이터: {len(val_images)}개로 분리.")