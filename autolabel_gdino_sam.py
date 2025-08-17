import os, glob, cv2, torch, numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# --- Config (change if needed) ---
UNLABELED_DIR = "data/unlabeled"
OUT_IMG_DIR   = "data/images/train"   # autolabeled go straight to train
OUT_LBL_DIR   = "data/labels/train"
PROMPTS = ["window", "door", "chimney", "skylight", "roof edge"]
CLASS_MAP = {"window":0, "door":1, "chimney":2, "skylight":3, "roof edge":4}
BOX_THR = 0.25
TEXT_THR = 0.25
SAM_CKPT = "models/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Grounding DINO (HuggingFace tiny)
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino = AutoModelForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE).eval()

# SAM
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.to(DEVICE)
predictor = SamPredictor(sam)

def xyxy_to_yolo(x1,y1,x2,y2, W,H):
    cx = (x1+x2)/2.0 / W
    cy = (y1+y2)/2.0 / H
    bw = (x2-x1) / W
    bh = (y2-y1) / H
    return cx,cy,bw,bh

for img_path in glob.glob(os.path.join(UNLABELED_DIR, "*.*")):
    img = cv2.imread(img_path)
    if img is None: 
        continue
    H, W = img.shape[:2]

    # DINO (multi-prompt)
    inputs = processor(images=img[:,:,::-1], text=". ".join(PROMPTS), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gdino(**inputs)
    target_sizes = torch.tensor([[H,W]]).to(DEVICE)
    res = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target_sizes, box_threshold=BOX_THR, text_threshold=TEXT_THR
    )[0]

    # SAM refine (optional; YOLO still needs bbox)
    predictor.set_image(img[:,:,::-1])
    stem = Path(img_path).stem
    out_img = os.path.join(OUT_IMG_DIR, f"{stem}.jpg")
    out_lbl = os.path.join(OUT_LBL_DIR, f"{stem}.txt")
    cv2.imwrite(out_img, img)
    lines=[]

    for box, lbl in zip(res["boxes"], res["labels"]):
        cls_name = lbl.lower()
        if "roof" in cls_name: cls_name = "roof edge"
        if cls_name not in CLASS_MAP: 
            continue
        x1,y1,x2,y2 = [float(v) for v in box.tolist()]
        # quick sanity filters
        if (x2-x1)*(y2-y1) < 600: 
            continue

        # SAM (optional)
        try:
            b = np.array([[x1,y1,x2,y2]])
            _m,_s,_ = predictor.predict(box=b, multimask_output=False)
        except Exception:
            pass

        cx, cy, bw, bh = xyxy_to_yolo(x1,y1,x2,y2,W,H)
        lines.append(f"{CLASS_MAP[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        with open(out_lbl,"w") as f: f.write("\n".join(lines))
    print(f"[AUTO] {stem}: {len(lines)} objects")
