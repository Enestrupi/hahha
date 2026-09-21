#!/usr/bin/env python3
"""
Batch processor for REAL book photos (on table) -> Clean ebook PDF like digital example

Usage:
  python batch_real_book.py --input ./real_photos --output MATEMATIKA_II_clean.pdf

Your case: Photos of physical book on wooden table with shadows
Output: Clean white pages like MATEMATIKA digital example
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')
from scanner_server import process_image_full

def process_folder(input_path, output_pdf, title="MATEMATIKA II"):
    input_path = Path(input_path)
    images = []
    for ext in ('*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG'):
        images.extend(input_path.glob(ext))
    images = sorted(images)
    
    if not images:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(images)} real book photos")
    print(f"Processing to remove background, shadows, fix perspective...")
    
    cleaned = []
    for idx, p in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {p.name}...", end=' ')
        img = cv2.imread(str(p))
        if img is None:
            print("FAILED")
            continue
        result, pts = process_image_full(img)
        cleaned.append(result)
        print(f"OK - detected={pts is not None} -> {result.shape}")
    
    if not cleaned:
        print("No images processed")
        return
    
    print(f"\nSaving clean ebook PDF to {output_pdf}...")
    pil_images=[]
    for img in cleaned:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pil_images[0].save(
        output_path,
        "PDF",
        save_all=True,
        append_images=pil_images[1:],
        resolution=200,
        quality=90,
        title=title
    )
    
    size_mb = output_path.stat().st_size / 1024/1024
    print(f"✓ Saved clean ebook: {output_path} ({size_mb:.2f} MB, {len(cleaned)} pages)")
    print(f"  Looks like your digital example - white background, no table!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real book photos -> Clean ebook")
    parser.add_argument('--input', '-i', required=True, help='Folder with real book photos (on table)')
    parser.add_argument('--output', '-o', default='MATEMATIKA_II_clean_ebook.pdf', help='Output PDF')
    parser.add_argument('--title', default='MATEMATIKA II - tekst shkollor per vitin e dyte te gjimnazit', help='Ebook title')
    args = parser.parse_args()
    process_folder(args.input, args.output, title=args.title)
