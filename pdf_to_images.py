import os, sys, fitz  # PyMuPDF
from pathlib import Path

def extract_pdf(pdf_path: str, out_dir: str, dpi: int = 300):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    seen = set()
    saved = 0

    # A) Lossless: extract embedded bitmaps
    for pno in range(len(doc)):
        for img in doc.get_page_images(pno, full=True):
            xref = img[0]
            if xref in seen: 
                continue
            seen.add(xref)
            base = doc.extract_image(xref)
            ext = base["ext"]  # png/jpg/jp2/…
            name = f"{Path(pdf_path).stem}_p{pno+1:03}_xref{xref}.{ext}"
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(base["image"])
            saved += 1

    # B) Fallback: render page and crop image blocks (captures placed photos)
    for pno in range(len(doc)):
        page = doc[pno]
        raw = page.get_text("rawdict")
        blocks = [b for b in raw["blocks"] if b.get("type") == 1]  # 1 = image block
        if not blocks:
            continue
        for i, b in enumerate(blocks, 1):
            clip = fitz.Rect(*b["bbox"])
            pix = page.get_pixmap(dpi=dpi, clip=clip)
            name = f"{Path(pdf_path).stem}_p{pno+1:03}_block{i}.png"
            pix.save(os.path.join(out_dir, name))
            saved += 1

    doc.close()
    return saved

if __name__ == "__main__":
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "input_pdfs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/unlabeled"
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    total = 0
    for fn in os.listdir(pdf_dir):
        if fn.lower().endswith(".pdf"):
            n = extract_pdf(os.path.join(pdf_dir, fn), out_dir, dpi)
            print(f"[OK] {fn}: saved {n} images")
            total += n
    print(f"Done. Total images: {total} → {out_dir}")
