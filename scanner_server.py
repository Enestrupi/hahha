#!/usr/bin/env python3
"""
AI Document Scanner Server v3 - Fixed to preserve text, remove background properly
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def shrink_quad(pts, factor=0.04):
    """Shrink quad towards center to remove background border"""
    center = np.mean(pts, axis=0)
    shrunk = []
    for p in pts:
        # Move point towards center by factor
        new_p = p + (center - p) * factor
        shrunk.append(new_p)
    return np.array(shrunk, dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    maxWidth = max(maxWidth, 100)
    maxHeight = max(maxHeight, 100)
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_page_smart(image):
    h,w = image.shape[:2]
    small = cv2.resize(image, (800, int(800*h/w)))
    ratio_h = h / small.shape[0]
    ratio_w = w / small.shape[1]
    
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv2.bitwise_or(mask_white, mask_red)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < small.shape[0]*small.shape[1]*0.08:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) >=4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            w_rect, h_rect = rect[1]
            if w_rect==0 or h_rect==0:
                continue
            aspect = min(w_rect,h_rect)/max(w_rect,h_rect)
            if aspect < 0.3:
                continue
            box = box.astype(float)
            box[:,0] = np.clip(box[:,0] * ratio_w, 0, w)
            box[:,1] = np.clip(box[:,1] * ratio_h, 0, h)
            # Shrink to remove background
            box = shrink_quad(box, 0.04)
            return box
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(approx) > small.shape[0]*small.shape[1]*0.15:
            pts = approx.reshape(4,2).astype(float)
            pts[:,0] *= ratio_w
            pts[:,1] *= ratio_h
            pts = shrink_quad(pts, 0.03)
            return pts
    return None

def refine_crop_to_page(image):
    """After initial warp, refine crop to just white page content"""
    # Detect white page inside warped image
    h,w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 55, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find largest white contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < h*w*0.3:
        return image
    
    x,y,w_box,h_box = cv2.boundingRect(largest)
    # Add small margin
    margin = 10
    x = max(0, x-margin)
    y = max(0, y-margin)
    w_box = min(w - x, w_box + 2*margin)
    h_box = min(h - y, h_box + 2*margin)
    
    # Only crop if it makes sense (not too small)
    if w_box > w*0.5 and h_box > h*0.5:
        return image[y:y+h_box, x:x+w_box]
    return image

def clean_to_ebook_style_v3(image):
    """Improved cleaning that preserves text perfectly"""
    # Check if red cover
    hsv_check = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_check, np.array([0,70,50]), np.array([10,255,255])) | cv2.inRange(hsv_check, np.array([170,70,50]), np.array([180,255,255]))
    red_ratio = np.sum(red_mask>0) / (image.shape[0]*image.shape[1])
    is_red_cover = red_ratio > 0.25
    
    if is_red_cover:
        # For red cover, just enhance slightly and keep as is
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l,a,b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Sharpen
        gaussian = cv2.GaussianBlur(result, (0,0), 1)
        result = cv2.addWeighted(result, 1.3, gaussian, -0.3, 0)
        return result
    
    # For white pages: advanced shadow removal
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    
    # Estimate background illumination with large blur
    blur = cv2.GaussianBlur(l, (0,0), sigmaX=40, sigmaY=40)
    # Division method to remove shadows
    divided = cv2.divide(l, blur, scale=255)
    # Blend to keep natural look
    l_corrected = cv2.addWeighted(divided, 0.6, l, 0.4, 0)
    
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_corrected = clahe.apply(l_corrected)
    
    lab_corrected = cv2.merge([l_corrected, a, b])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    # Gentle sharpen
    gaussian = cv2.GaussianBlur(result, (0,0), 1.2)
    result = cv2.addWeighted(result, 1.25, gaussian, -0.25, 0)
    
    # Make paper whiter - but preserve text
    # Only whiten where it's already very bright and low saturation (paper)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_bg = np.array([0, 0, 200])
    upper_bg = np.array([180, 35, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Erode to avoid eating text
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)
    
    # For background, boost to white but keep some texture
    # Instead of pure white, make it 245+ for natural paper
    result[bg_mask==255] = [255,255,255]
    
    return result

def process_image_full(image):
    pts = detect_page_smart(image)
    if pts is not None:
        try:
            warped = four_point_transform(image, pts)
            # Refine crop
            warped = refine_crop_to_page(warped)
        except Exception as e:
            print(f"Warp failed: {e}")
            warped = image
            pts = None
    else:
        warped = image
    
    cleaned = clean_to_ebook_style_v3(warped)
    return cleaned, pts

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    cleaned, pts = process_image_full(img)
    _, buffer = cv2.imencode('.jpg', cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buffer).decode('utf-8')
    points_list = pts.tolist() if pts is not None else None
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{b64}',
        'points': points_list,
        'width': cleaned.shape[1],
        'height': cleaned.shape[0],
        'detected': pts is not None
    })

@app.route('/batch', methods=['POST'])
def batch_endpoint():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images'}), 400
    cleaned_images = []
    for f in files:
        nparr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cleaned, _ = process_image_full(img)
        cleaned_images.append(cleaned)
    
    if not cleaned_images:
        return jsonify({'error': 'No valid images'}), 400
    
    from PIL import Image as PILImage
    pil_images = []
    for img in cleaned_images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        pil_images.append(pil)
    
    import io
    pdf_buffer = io.BytesIO()
    pil_images[0].save(pdf_buffer, "PDF", save_all=True, append_images=pil_images[1:], resolution=200, quality=90)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name='MATEMATIKA_II_clean_ebook.pdf')

@app.route('/')
def index():
    return "AI Scanner Server v3 - Fixed text preservation"

if __name__ == '__main__':
    print("Starting AI Scanner Server v3 on port 8001...")
    app.run(host='0.0.0.0', port=8001, debug=False)
