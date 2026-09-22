#!/usr/bin/env python3
"""
Genius AI Server - Fully automatic, does everything on its own
- Auto-detects page
- Auto-rotates crooked photos
- Auto-removes background (wooden table)
- Auto-cleans shadows
- No manual needed
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import math

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

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_rotate_image(image):
    """Genius auto-rotation - detects text orientation and straightens"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Detect text lines via Hough transform
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    angles=[]
    if lines is not None:
        for rho, theta in lines[:,0]:
            # Convert theta to degrees, text lines should be near 0 or 180 degrees (horizontal)
            angle = (theta * 180 / np.pi) - 90
            # Only consider near-horizontal lines (text)
            if abs(angle) < 30 or abs(angle) > 150:
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                if abs(angle) < 30:
                    angles.append(angle)
    
    if angles:
        # Median angle is the skew
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5 and abs(median_angle) < 30:
            # Rotate to straighten
            (h,w) = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated, median_angle
    
    # Method 2: Try 0, 90, 180, 270 rotations and check which has most horizontal text
    # Use projection profile method
    best_angle=0
    best_score=0
    for test_angle in [0, 90, 180, 270]:
        if test_angle==0:
            test_img=gray
        else:
            (h,w)=gray.shape[:2]
            center=(w//2,h//2)
            M=cv2.getRotationMatrix2D(center, test_angle, 1.0)
            test_img=cv2.warpAffine(gray, M, (w,h))
        
        # Check horizontal projection variance - text should have high variance
        # Sum rows, variance should be high for horizontal text
        hist = np.sum(test_img < 200, axis=1)  # Count dark pixels per row
        score = np.var(hist)
        if score > best_score:
            best_score=score
            best_angle=test_angle
    
    if best_angle!=0:
        (h,w)=image.shape[:2]
        center=(w//2,h//2)
        M=cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated=cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, best_angle
    
    return image, 0

def detect_page_genius(image):
    """Genius page detection - finds white page even with wooden table background"""
    h,w = image.shape[:2]
    # Downscale for speed but keep enough detail
    small = cv2.resize(image, (1000, int(1000*h/w)))
    rh = h / small.shape[0]
    rw = w / small.shape[1]
    
    # Multiple detection methods combined
    
    # Method 1: White color detection in HSV (most reliable for book pages)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Method 2: Red cover detection
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Method 3: High brightness detection
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, mask_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    mask_combined = cv2.bitwise_or(mask_white, mask_red)
    mask_combined = cv2.bitwise_or(mask_combined, mask_bright)
    
    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours,_ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < small.shape[0]*small.shape[1]*0.08:
            continue
        
        # Get bounding quadrilateral
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        w_rect, h_rect = rect[1]
        if w_rect==0 or h_rect==0:
            continue
        
        aspect = min(w_rect,h_rect)/max(w_rect,h_rect)
        if aspect < 0.25:  # Too thin, not a page
            continue
        
        # Check if quad looks like page (should be roughly rectangular, not too skewed)
        box = box.astype(float)
        box[:,0] *= rw
        box[:,1] *= rh
        
        # Shrink slightly to remove border/background
        center = np.mean(box, axis=0)
        box = box + (center - box)*0.05  # Shrink 5%
        
        # Clip to image bounds
        box[:,0] = np.clip(box[:,0], 0, w)
        box[:,1] = np.clip(box[:,1], 0, h)
        
        return box
    
    # Fallback: edge detection
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.GaussianBlur(gray_small, (5,5), 0)
    edged = cv2.Canny(gray_small, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    contours,_ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(approx) > small.shape[0]*small.shape[1]*0.15:
            pts = approx.reshape(4,2).astype(float)
            pts[:,0] *= rw
            pts[:,1] *= rh
            center = np.mean(pts, axis=0)
            pts = pts + (center - pts)*0.05
            return pts
    
    return None

def clean_genius(image):
    """Genius cleaning - removes shadows, makes white background, preserves text perfectly"""
    # Check if red cover
    hsv_check = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_check, np.array([0,70,50]), np.array([10,255,255])) | cv2.inRange(hsv_check, np.array([170,70,50]), np.array([180,255,255]))
    red_ratio = np.sum(red_mask>0) / (image.shape[0]*image.shape[1])
    is_red = red_ratio > 0.25
    
    if is_red:
        # For red cover, just enhance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l,a,b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gaussian = cv2.GaussianBlur(result, (0,0), 1)
        result = cv2.addWeighted(result, 1.3, gaussian, -0.3, 0)
        return result
    
    # For white pages: genius shadow removal
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    
    # Large blur to estimate illumination
    blur = cv2.GaussianBlur(l, (0,0), sigmaX=50, sigmaY=50)
    
    # Division method - standard for document scanning, removes shadows
    divided = cv2.divide(l, blur, scale=255)
    
    # Blend original and divided for natural look
    l_corr = cv2.addWeighted(divided, 0.65, l, 0.35, 0)
    
    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    l_corr = clahe.apply(l_corr)
    
    lab_corr = cv2.merge([l_corr, a, b])
    result = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
    
    # Gentle sharpen
    gaussian = cv2.GaussianBlur(result, (0,0), 1.0)
    result = cv2.addWeighted(result, 1.3, gaussian, -0.3, 0)
    
    # Make paper white - but preserve text
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_bg = np.array([0, 0, 195])
    upper_bg = np.array([180, 40, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Erode to avoid eating text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)
    
    # For background, set to pure white
    result[bg_mask==255] = [255,255,255]
    
    return result

def process_genius(image):
    """Full genius pipeline: auto-rotate + detect + clean - does everything on its own"""
    # Step 1: Auto-rotate crooked photos
    rotated, angle = auto_rotate_image(image)
    
    # Step 2: Detect page and remove background
    pts = detect_page_genius(rotated)
    if pts is not None:
        try:
            warped = four_point_transform(rotated, pts)
        except:
            warped = rotated
            pts = None
    else:
        warped = rotated
    
    # Step 3: Clean shadows and make white background
    cleaned = clean_genius(warped)
    
    return cleaned, pts, angle

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    cleaned, pts, angle = process_genius(img)
    _, buffer = cv2.imencode('.jpg', cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{b64}',
        'points': pts.tolist() if pts is not None else None,
        'angle': float(angle),
        'width': cleaned.shape[1],
        'height': cleaned.shape[0],
        'detected': pts is not None
    })

@app.route('/batch', methods=['POST'])
def batch_endpoint():
    """Batch process 316 photos at once - fully automatic"""
    files = request.files.getlist('images')
    if not files:
        # Try single file upload with multiple
        files = []
        for key in request.files:
            files.extend(request.files.getlist(key))
    
    if not files:
        return jsonify({'error': 'No images'}), 400
    
    cleaned_images=[]
    results=[]
    
    for idx, f in enumerate(files):
        nparr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        cleaned, pts, angle = process_genius(img)
        cleaned_images.append(cleaned)
        results.append({
            'name': f.filename,
            'detected': pts is not None,
            'angle': float(angle),
            'width': cleaned.shape[1],
            'height': cleaned.shape[0]
        })
    
    if not cleaned_images:
        return jsonify({'error': 'No valid images'}), 400
    
    # Create PDF
    pil_images=[]
    for img in cleaned_images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    
    pdf_buffer = io.BytesIO()
    pil_images[0].save(pdf_buffer, "PDF", save_all=True, append_images=pil_images[1:], resolution=200, quality=85)
    pdf_buffer.seek(0)
    
    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True, download_name=f'MATEMATIKA_II_{len(cleaned_images)}_pages_genius.pdf')

@app.route('/genius', methods=['POST'])
def genius_endpoint():
    """Genius endpoint - upload 1 photo, get back clean ebook page, fully automatic"""
    return scan_endpoint()

@app.route('/')
def index():
    return """
    <h1>🧠 Genius AI Server - Fully Automatic</h1>
    <p>Does everything on its own:</p>
    <ul>
      <li>✅ Auto-rotates crooked photos</li>
      <li>✅ Auto-detects page, removes wooden table background</li>
      <li>✅ Auto-cleans shadows, makes white background</li>
      <li>✅ No manual needed</li>
    </ul>
    <p>Endpoints:</p>
    <ul>
      <li>POST /scan - Single image → clean page</li>
      <li>POST /batch - Multiple images → PDF ebook (handles 316 at once)</li>
    </ul>
    """

if __name__ == '__main__':
    print("🧠 Starting Genius AI Server on port 8001...")
    print("   Does everything automatically: rotate, remove background, clean")
    app.run(host='0.0.0.0', port=8001, debug=False)
