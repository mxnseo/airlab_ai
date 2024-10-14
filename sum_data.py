'''
import os
import shutil
import random

# 기본 경로 설정
base_path = 'D:/ChanWooKim/dataset/test'
output_folder = os.path.join(base_path, 'all')

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 각 폴더를 순회
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    img_folder_path = os.path.join(folder_path, 'img')

    # img 폴더가 존재하는 경우
    if os.path.isdir(img_folder_path):
        for img_file in os.listdir(img_folder_path):
            if img_file.endswith('.png'):
                # 0.1 확률로 파일 선택
                if random.random() < 0.1:
                    src_file = os.path.join(img_folder_path, img_file)
                    dst_file = os.path.join(output_folder, img_file)
                    shutil.copy(src_file, dst_file)
                    print(f'복사됨: {src_file} -> {dst_file}')

'''

import os
import random

# 이미지 파일이 있는 폴더 경로 설정
folder_path = 'D:/ChanWooKim/dataset/test_all/'  # 수정 필요

# 폴더 내의 모든 파일을 순회
for img_file in os.listdir(folder_path):
    # 파일이 PNG 형식인 경우만 처리
    if img_file.endswith('.png'):
        # 0.5 확률로 파일 삭제
        if random.random() < 0.5:
            file_path = os.path.join(folder_path, img_file)
            os.remove(file_path)
            print(f'삭제됨: {file_path}')
