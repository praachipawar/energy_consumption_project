import os, random, shutil
from pathlib import Path

IMG_DIR = "data/images/train"
LBL_DIR = "data/labels/train"
VAL_FRAC = 0.2

out_img_train = Path("data/images/train")
out_lbl_train = Path("data/labels/train")
out_img_val = Path("data/images/val")
out_lbl_val = Path("data/labels/val")
for p in [out_img_val, out_lbl_val]:
    p.mkdir(parents=True, exist_ok=True)

imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
random.shuffle(imgs)
cut = int(len(imgs)*(1-VAL_FRAC))
train_imgs, val_imgs = imgs[:cut], imgs[cut:]

def move_set(files, tgt_img_dir, tgt_lbl_dir):
    for f in files:
        stem = Path(f).stem
        lbl = stem + ".txt"
        shutil.copy2(os.path.join(IMG_DIR,f), os.path.join(tgt_img_dir,f))
        if os.path.exists(os.path.join(LBL_DIR,lbl)):
            shutil.copy2(os.path.join(LBL_DIR,lbl), os.path.join(tgt_lbl_dir,lbl))

move_set(val_imgs, out_img_val, out_lbl_val)
print(f"Moved {len(val_imgs)} images to val/")
