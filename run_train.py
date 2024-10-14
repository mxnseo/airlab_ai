from ultralytics import YOLO
import multiprocessing
from ultralytics import settings

'''
import multiprocessing

multiprocessing.freeze_support()

model = YOLO("yolov10s-seg.yaml")
# Train the model with 2 GPUs
# device는 GPU개수로 테스트 환경에서는 RTX4090 8개로 측정함.
# Compete_segment.yaml은 cfg.datasets에 있고
# 배치 크기 8 
results = model.train(data="Compete_segment.yaml",pretrained='yolov8s-seg.pt',epochs=30, device=[0], workers=0, batch=8)
'''

'''
if __name__=="__main__":
    settings.reset()
    multiprocessing.freeze_support()

    model = YOLO("yolov10s-seg.yaml")
    # 배치 크기 8 
    # multiprocession.freeze_support()을 쓰지 않는다면 num_worker 0으로 해야함.
    # num_worker 4로 설정해주니 GPU를 100%까지도 활용하는 것을 확인함.
    results = model.train(data="Compete_segment.yaml", pretrained='./yolov11x-seg.pt',epochs=100, device=[0], workers=4, batch=8)
'''
    

if __name__=="__main__":
    settings.reset()
    multiprocessing.freeze_support()

    model = YOLO("yolov10s-seg.yaml")
    results = model.train(data="Compete_segment.yaml", pretrained='D:/ChanWooKim/auto_dna/runs/100_Epoch_edit_parameter(10.13)/weights/best.pt', epochs=100, device=[0], workers=4, batch=8)