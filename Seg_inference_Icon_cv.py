import numpy as np
import glob, cv2
import time
import random
import os
import torch
from ultralytics import YOLO

# YOLO 모델 경로 설정
model = YOLO('D:/ChanWooKim/auto_dna/runs/100+100Epoch_edit_parameter(10.14)/weights/best.pt').cuda()

# 클래스 설정
agent_classes = ['Car', 'Bus']
loc_classes = ['VehLane', 'OutgoLane', 'IncomLane', 'Jun', 'Parking']
action_classes = ['Brake', 'IncatLft', 'IncatRht', 'HazLit']
class_nums = [len(agent_classes), len(loc_classes), len(action_classes)]

# 아이콘 로드
icons = {}
for actions in action_classes + loc_classes:
    target = f'./Icons/{actions}.png'
    icon_img = cv2.imread(target)
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icons[actions] = icon_img

def seg_plot_one_box(x, idx, img, mask, cls, loc, action, color=None, line_thickness=None):
    tl = line_thickness or 2  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (np.clip(int(x[0]), 0, img.shape[1]), np.clip(int(x[1]), 0, img.shape[0])), (np.clip(int(x[2]), 0, img.shape[1]), np.clip(int(x[3]), 0, img.shape[0]))
    
    cv2.rectangle(img, c1, c2, color, thickness=1)

    num_icon = np.sum(action)
    icon_size = int(np.min([(c2[0] - c1[0]) / num_icon, (x[3] - x[1]) / 2, 64]))
    c3 = c1[0]

    try:
        offset_icon = 0
        for ii in range(len(action)):
            if action[ii] == 1:
                img[c1[1]:c1[1] + icon_size, c3 + offset_icon:c3 + offset_icon + icon_size, :] = \
                    cv2.resize(icons[action_classes[ii]], (icon_size, icon_size), interpolation=cv2.INTER_NEAREST) * 0.5 + img[c1[1]:c1[1] + icon_size, c3 + offset_icon:c3 + offset_icon + icon_size, :] * 0.5
                offset_icon += icon_size

        img[c2[1] - icon_size:c2[1], c3:c3 + icon_size, :] = \
            cv2.resize(icons[loc_classes[loc]], (icon_size, icon_size)) * 0.5 + img[c2[1] - icon_size:c2[1], c3:c3 + icon_size, :] * 0.5

    except Exception as e:
        print(f"Error in icon drawing: {e}")
    
    mask = mask[c1[1]:c2[1], c1[0]:c2[0]]
    mask = mask > 0.5
    img[c1[1]:c2[1], c1[0]:c2[0], :][mask] = (
        img[c1[1]:c2[1], c1[0]:c2[0], :][mask] * 0.65 + np.array(color) * 0.35
    )

# 결과 폴더 설정
result_folder = './Result/'
os.makedirs(result_folder, exist_ok=True)

target_folder = 'D:/ChanWooKim/dataset/test_all'  # Target Dir
searchLabel = sorted(os.listdir(target_folder))

with torch.no_grad():
    for jj in range(len(searchLabel)):
        img_name = os.path.join(target_folder, searchLabel[jj])

        results = model(img_name, verbose=False, imgsz=[480, 1280])

        target_outputs = results[0].boxes.data.cpu().numpy()
        target_img = results[0].orig_img
        target_img = np.array(target_img[:, :, ::-1])

        # 원본 이미지 읽기
        input_img = cv2.imread(img_name)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)  # BGR to RGB

        xyxy = target_outputs[:, 0:4]
        cls = target_outputs[:, 5].astype('int')
        loc = target_outputs[:, 6].astype('int')
        action = target_outputs[:, 7:].astype('int')

        for i in range(xyxy.shape[0]):
            seg_plot_one_box(xyxy[i], i, target_img, results[0].masks.data[i].cpu().numpy(), cls[i], loc[i], action[i])
            print(cls[i], end=' ')

        print('next')

        # OpenCV로 이미지 출력
        cv2.imshow('Input', input_img)
        cv2.imshow('Output', target_img)
        key = cv2.waitKey(0)  # 키 입력을 기다림, 0은 무한 대기
        if key == 27:  # ESC 키를 누르면 종료
            break

cv2.destroyAllWindows()
