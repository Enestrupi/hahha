import cv2
import numpy as np

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

def detect_page_improved(image):
    h,w = image.shape[:2]
    # Downscale for speed
    small = cv2.resize(image, (800, int(800*h/w)))
    ratio_h = h / small.shape[0]
    ratio_w = w / small.shape[1]
    
    # Method 1: White color detection in HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Method 2: Edge detection
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    
    # Combine
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    mask_white = cv2.dilate(mask_white, kernel, iterations=1)
    
    # Find contours in white mask
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < small.shape[0]*small.shape[1]*0.1:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(float)
            pts[:,0] *= ratio_w
            pts[:,1] *= ratio_h
            return pts
        # Try minAreaRect for non-perfect quads
        if len(approx) >=4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(float)
            box[:,0] *= ratio_w
            box[:,1] *= ratio_h
            return box
    
    # Method 3: Edge based fallback
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and cv2.contourArea(approx) > small.shape[0]*small.shape[1]*0.15:
            pts = approx.reshape(4,2).astype(float)
            pts[:,0] *= ratio_w
            pts[:,1] *= ratio_h
            return pts
    
    return None

def clean_image_advanced(image):
    # Shadow removal + white background
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Illumination correction via large blur division
    # Split and process each channel
    result_planes=[]
    for plane in cv2.split(result):
        # Estimate background with large gaussian
        bg = cv2.GaussianBlur(plane, (0,0), sigmaX=50, sigmaY=50)
        # Avoid division by zero
        # Normalize: result = plane * (mean(bg)/bg)
        mean_bg = np.mean(bg)
        # Divide
        # Use float
        plane_f = plane.astype(float)
        bg_f = bg.astype(float)
        bg_f[bg_f==0]=1
        corrected = plane_f * (mean_bg / bg_f)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        result_planes.append(corrected)
    result = cv2.merge(result_planes)
    
    # Make background whiter
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Threshold to find near-white background
    _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    # Dilate mask slightly
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Set background to white
    result[mask==255] = [255,255,255]
    
    # Sharpen
    # Unsharp mask
    gaussian = cv2.GaussianBlur(result, (0,0), 3)
    result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    
    return result

for name in ["../uploads/IMG_1215.jpeg", "../uploads/IMG_1216.jpeg", "../uploads/IMG_1217.jpeg"]:
    img = cv2.imread(name)
    print(f"\n{name}: {img.shape}")
    pts = detect_page_improved(img)
    print(f"  Detected pts: {pts}")
    if pts is not None:
        warped = four_point_transform(img, pts)
        print(f"  Warped: {warped.shape}")
        cleaned = clean_image_advanced(warped)
        cv2.imwrite(f"improved_{name.split('/')[-1]}", cleaned)
        print(f"  Saved improved_{name.split('/')[-1]}")
    else:
        cleaned = clean_image_advanced(img)
        cv2.imwrite(f"improved_{name.split('/')[-1]}", cleaned)
        print(f"  No pts, cleaned full image")
