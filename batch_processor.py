#!/usr/bin/env python3
"""
AI Ebook Maker - Batch Processor
For 316+ tall screenshots like the example (3 of 226)

Features:
- Auto-crop black borders (from screenshot viewer)
- Auto-split tall vertical stacks into single pages
- Enhance: contrast, sharpen, white balance
- Deduplication
- Export to PDF ebook

Usage:
    pip install Pillow
    python batch_processor.py --input ./photos --output ebook.pdf --auto-split --enhance --title "MATEMATIKA VIII II"

Your example image: one tall PNG containing 5 pages stacked vertically with black background
-> This script will detect white gaps / black separator lines and split into 5 pages.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io

def is_black_pixel(r,g,b, thresh=30):
    return r < thresh and g < thresh and b < thresh

def auto_crop_black_border(img, margin=10):
    """Crop black borders from screenshot like your example"""
    w, h = img.size
    # Convert to grayscale for analysis, downsample for speed
    small = img.resize((400, int(400*h/w)), Image.LANCZOS)
    pix = small.load()
    sw, sh = small.size
    
    # Find bounding box of non-black
    top, bottom, left, right = 0, sh-1, 0, sw-1
    
    # top
    for y in range(sh):
        for x in range(0, sw, 5):
            r,g,b = pix[x,y][:3] if len(pix[x,y])>3 else pix[x,y]
            if not is_black_pixel(r,g,b):
                top = y
                break
        else:
            continue
        break
    # bottom
    for y in range(sh-1, -1, -1):
        for x in range(0, sw, 5):
            r,g,b = pix[x,y][:3] if len(pix[x,y])>3 else pix[x,y]
            if not is_black_pixel(r,g,b):
                bottom = y
                break
        else:
            continue
        break
    # left
    for x in range(sw):
        for y in range(top, bottom, 5):
            r,g,b = pix[x,y][:3] if len(pix[x,y])>3 else pix[x,y]
            if not is_black_pixel(r,g,b):
                left = x
                break
        else:
            continue
        break
    # right
    for x in range(sw-1, -1, -1):
        for y in range(top, bottom, 5):
            r,g,b = pix[x,y][:3] if len(pix[x,y])>3 else pix[x,y]
            if not is_black_pixel(r,g,b):
                right = x
                break
        else:
            continue
        break
    
    # Convert back to original scale
    scale_x = w / sw
    scale_y = h / sh
    l = max(0, int(left*scale_x - margin))
    t = max(0, int(top*scale_y - margin))
    r = min(w, int(right*scale_x + margin))
    b = min(h, int(bottom*scale_y + margin))
    
    if (r-l) < w*0.5 or (b-t) < h*0.5:
        return img  # avoid over-crop
    return img.crop((l,t,r,b))

def detect_splits(img):
    """Detect horizontal gaps between pages"""
    w, h = img.size
    if h < w*1.4:
        return [img]  # not tall enough
    
    # Downscale for analysis
    aw = 400
    ah = int(aw * h / w)
    small = img.resize((aw, ah)).convert('RGB')
    pixels = list(small.getdata())
    # Reshape to rows
    row_stats = []
    for y in range(ah):
        white = 0
        black = 0
        bright_sum = 0
        for x in range(aw):
            r,g,b = pixels[y*aw + x]
            bright = (r+g+b)//3
            bright_sum += bright
            if bright > 230:
                white+=1
            if bright < 30:
                black+=1
        row_stats.append({
            'white_ratio': white/aw,
            'black_ratio': black/aw,
            'avg': bright_sum/aw
        })
    
    gaps=[]
    in_gap=False
    gap_start=0
    for y, s in enumerate(row_stats):
        is_gap = s['white_ratio']>0.88 or s['black_ratio']>0.7 or s['avg']>240
        if is_gap and not in_gap:
            in_gap=True
            gap_start=y
        elif not is_gap and in_gap:
            in_gap=False
            gap_end=y
            gh = gap_end-gap_start
            if gh>12 and gap_start>20 and gap_end < ah-20:
                gaps.append((gap_start+gap_end)//2)
    
    # Filter close gaps
    filtered=[]
    for g in gaps:
        if not filtered or g - filtered[-1] > 40:
            filtered.append(g)
    
    if not filtered:
        return [img]
    
    # Convert to original coordinates
    split_ys = [int(g * h / ah) for g in filtered]
    
    # Create splits
    result=[]
    last=0
    for y in split_ys:
        if y - last > w*0.6:
            result.append(img.crop((0, last, w, y)))
            last=y
    if h - last > w*0.6:
        result.append(img.crop((0, last, w, h)))
    
    return result if len(result)>1 else [img]

def enhance_image(img):
    """Auto enhance: contrast, brightness, sharpen"""
    # Auto contrast
    img = ImageOps.autocontrast(img, cutoff=0.5)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)
    # Sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    return img

def process_folder(input_path, auto_split=True, auto_crop=True, enhance=True, dedup=True):
    input_path = Path(input_path)
    images = []
    for ext in ('*.png','*.jpg','*.jpeg','*.webp','*.PNG','*.JPG','*.JPEG'):
        images.extend(input_path.glob(ext))
    images = sorted(images)
    
    if not images:
        print(f"No images found in {input_path}")
        return []
    
    print(f"Found {len(images)} input files")
    all_pages=[]
    seen_hashes=set()
    
    for idx, img_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] Processing {img_path.name}...", end=' ')
        try:
            img = Image.open(img_path).convert('RGB')
            if auto_crop:
                img = auto_crop_black_border(img)
            
            splits = detect_splits(img) if auto_split else [img]
            
            for s in splits:
                if enhance:
                    s = enhance_image(s)
                # Dedup via simple hash
                if dedup:
                    # downscale to 16x16 hash
                    h = s.resize((16,16), Image.LANCZOS).tobytes()
                    if h in seen_hashes:
                        print(f" duplicate skipped", end=' ')
                        continue
                    seen_hashes.add(h)
                all_pages.append(s)
            print(f"-> {len(splits)} page(s) | total: {len(all_pages)}")
        except Exception as e:
            print(f" ERROR: {e}")
    
    return all_pages

def save_pdf(pages, output_path, title="Ebook", author="AI Ebook Maker", quality=90, page_size="A4"):
    from PIL import Image
    if not pages:
        print("No pages to save")
        return
    
    # Page size in pixels at 150 DPI
    sizes = {
        "A4": (1240, 1754), # 210x297mm at 150dpi
        "A5": (874, 1240),
        "LETTER": (1275, 1650),
        "ORIGINAL": None
    }
    
    target = sizes.get(page_size, sizes["A4"])
    
    print(f"Saving PDF to {output_path} with {len(pages)} pages...")
    
    # Convert to PDF using Pillow (first page saves with append)
    # For better quality, we save each page centered on white background if needed
    
    processed=[]
    for p in pages:
        if target is None:
            processed.append(p)
        else:
            tw, th = target
            # Fit image inside target with margins
            margin=40
            max_w = tw - margin*2
            max_h = th - margin*2
            pw, ph = p.size
            scale = min(max_w/pw, max_h/ph, 1.0)
            nw, nh = int(pw*scale), int(ph*scale)
            resized = p.resize((nw, nh), Image.LANCZOS)
            # Create white background
            bg = Image.new('RGB', (tw, th), 'white')
            bg.paste(resized, ((tw-nw)//2, (th-nh)//2))
            processed.append(bg)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pillow PDF save
    processed[0].save(
        output_path,
        "PDF",
        resolution=150,
        save_all=True,
        append_images=processed[1:],
        quality=quality,
        title=title,
        author=author
    )
    
    size_mb = output_path.stat().st_size / 1024/1024
    print(f"✓ Saved {output_path} ({size_mb:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="AI Ebook Maker - Convert 316 photos to PDF ebook")
    parser.add_argument('--input', '-i', required=True, help='Input folder with photos')
    parser.add_argument('--output', '-o', default='ebook.pdf', help='Output PDF path')
    parser.add_argument('--auto-split', action='store_true', default=True, help='Auto-split tall screenshots')
    parser.add_argument('--no-auto-split', dest='auto_split', action='store_false')
    parser.add_argument('--auto-crop', action='store_true', default=True)
    parser.add_argument('--no-auto-crop', dest='auto_crop', action='store_false')
    parser.add_argument('--enhance', action='store_true', default=True)
    parser.add_argument('--no-enhance', dest='enhance', action='store_false')
    parser.add_argument('--title', default='MATEMATIKA VIII II', help='Ebook title')
    parser.add_argument('--author', default='AI Ebook Maker', help='Author')
    parser.add_argument('--page-size', default='A4', choices=['A4','A5','LETTER','ORIGINAL'])
    parser.add_argument('--quality', type=int, default=90, help='JPEG quality 1-95')
    
    args = parser.parse_args()
    
    pages = process_folder(args.input, auto_split=args.auto_split, auto_crop=args.auto_crop, enhance=args.enhance)
    save_pdf(pages, args.output, title=args.title, author=args.author, quality=args.quality, page_size=args.page_size)

if __name__ == '__main__':
    main()
