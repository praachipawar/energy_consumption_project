from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # start light; upgrade to s/m once data grows

model.train(
    data="data.yaml",
    epochs=120,
    imgsz=640,
    batch=16,
    lr0=0.001,
    patience=20,
    weight_decay=0.0005,
    device=0,                 # 'cpu' if no GPU
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, degrees=10, translate=0.08, scale=0.5
)
