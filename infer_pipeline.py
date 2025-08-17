import cv2, torch, json, numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

YOLO_WEIGHTS = "runs/detect/train/weights/best.pt"  # or yolov8n.pt initially
IMG_PATH = "data/images/val/any_image.jpg"          # pick a sample
OUT_JSON = "det_output.json"
SAM_CKPT = "models/sam_vit_b_01ec64.pth"
CONF_THR = 0.35
FALLBACK_MIN_OBJS = 1
CLASS_NAMES = ["window","door","chimney","skylight","roof_edge"]

yolo = YOLO(YOLO_WEIGHTS)
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino = AutoModelForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda" if torch.cuda.is_available() else "cpu").eval()
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

def run_yolo(img):
    r = yolo(img, conf=0.25, verbose=False)[0]
    dets=[]
    if r.boxes is None: return dets
    for b in r.boxes:
        cls_id = int(b.cls.item())
        if 0 <= cls_id < len(CLASS_NAMES):
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            dets.append({"cls":CLASS_NAMES[cls_id], "conf":float(b.conf.item()), "box":[x1,y1,x2,y2]})
    return dets

def run_dino(img):
    H,W = img.shape[:2]
    q = "window. door. chimney. skylight. roof edge."
    inputs = processor(images=img[:,:,::-1], text=q, return_tensors="pt").to(gdino.device)
    with torch.no_grad():
        outputs = gdino(**inputs)
    target_sizes = torch.tensor([[H,W]]).to(gdino.device)
    res = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, box_threshold=0.25, text_threshold=0.25)[0]
    dets=[]
    for box, label in zip(res["boxes"], res["labels"]):
        cls = label.lower()
        if "roof" in cls: cls = "roof_edge"
        if cls not in CLASS_NAMES: continue
        x1,y1,x2,y2 = [float(v) for v in box.tolist()]
        dets.append({"cls":cls, "conf":0.30, "box":[x1,y1,x2,y2]})
    return dets

def add_sam_masks(img, dets):
    predictor.set_image(img[:,:,::-1])
    for d in dets:
        x1,y1,x2,y2 = d["box"]
        box = np.array([[x1,y1,x2,y2]])
        masks,scores,_ = predictor.predict(box=box, multimask_output=False)
        d["mask_score"] = float(scores[0])
    return dets

img = cv2.imread(IMG_PATH)
dets = run_yolo(img)
avg_conf = np.mean([d["conf"] for d in dets]) if dets else 0.0
if (len(dets) < FALLBACK_MIN_OBJS) or (avg_conf < CONF_THR):
    print("Low YOLO confidence → using Grounding DINO fallback.")
    dets = run_dino(img)

dets = add_sam_masks(img, dets)

with open(OUT_JSON,"w") as f: json.dump({"detections":dets}, f, indent=2)
print("Saved", OUT_JSON)
