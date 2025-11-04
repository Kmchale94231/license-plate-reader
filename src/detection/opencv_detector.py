import os
import numpy as np
import cv2
import kagglehub
import pytesseract as pyt

pyt.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

from matplotlib import pyplot as plt

dataset_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
img_path = os.path.join(dataset_path, "images", "Cars0.png")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at: {img_path}")

img = cv2.imread(img_path)
if img is None:
    raise RuntimeError(f"Could not read image: {img_path}")

def detect_candidates(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img_gray is not None, "File could not be read, check with os.path.exists()"

    img_blur = cv2.bilateralFilter(img_gray, 5, 25, 25)

    cv2.imshow('img_blur', img_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    edges = cv2.Canny(img_blur, 80, 240)

    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    contours, hierarchy = cv2.findContours(
        edges, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )             

    candidate_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        candidate_boxes.append((x, y, w, h))

    print("Contours found:", len(contours))
    print("Candidate boxes:", len(candidate_boxes))

    return contours

def filter_plate_candidates(contours, img, debug_samples=10):

    H, W = img.shape[:2]

    MIN_AREA = 0.0002 * W * H
    MAX_AREA = 0.60   * W * H

    AR_MIN, AR_MAX = 3.3, 8.0
    RECT_MIN = 0.60
    ANGLE_MAX = 35

    candidates = []

    c_short = c_zero = c_angle = 0 
    c_small = c_large = 0           
    c_ar = c_rect = 0                 
    c_pass = 0   

    for cnt in contours:
        if len(cnt) < 5:
            if c_short < debug_samples:
                print("[SKIP short] len(cnt) =", len(cnt))
            c_short += 1
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), ang = rect
        rw, rh = float(rw), float(rh)

        if rw <= 0 or rh <= 0:
            if c_zero < debug_samples:
                print("[SKIP zero] rw or rh <= 0  -> rw=%.1f rh=%.1f" % (rw, rh))
            c_zero += 1
            continue

        angle = ang
        if rw < rh:
            angle += 90.0
        if angle < -90.0:
            angle += 180.0
        elif angle > 90.0:
            angle -= 180.0

        if abs(angle) > ANGLE_MAX:
            if c_angle < debug_samples:
                print("[REJECT angle] angle=%.1f (> %d)" % (angle, ANGLE_MAX))
            c_angle += 1
            continue

        rot_area = rw * rh
        area_frac = rot_area / (W * H)

        if rot_area < MIN_AREA:
            if c_small < debug_samples:
                print("[REJECT small] rot_area=%.0f  area_frac=%.4f  (MIN=%.4f)" %
                      (rot_area, area_frac, MIN_AREA/(W*H)))
            c_small += 1
            continue

        if rot_area > MAX_AREA:
            if c_large < debug_samples:
                print("[REJECT large] rot_area=%.0f  area_frac=%.4f  (MAX=%.4f)" %
                      (rot_area, area_frac, MAX_AREA/(W*H)))
            c_large += 1
            continue

        short_side = min(rw, rh)
        if short_side < 6:
            if c_small < debug_samples:
                print("[REJECT thin] short_side=%.1f px" % short_side)
            c_small += 1
            continue

        ar = max(rw, rh) / min(rw, rh)
        if c_small + c_large < 15:
            print(f"rot_area={rot_area:.0f}  area_frac={area_frac:.4f}")
            
        if rot_area > MAX_AREA:
            c_large += 1
            continue 

        ar = max(rw, rh) / min(rw, rh)

        if ar < AR_MIN or ar > AR_MAX:
            if c_ar < debug_samples:
                print("[REJECT AR] ar=%.2f (range %.1f–%.1f)  rw=%.1f rh=%.1f  angle=%.1f" %
                      (ar, AR_MIN, AR_MAX, rw, rh, angle))
            c_ar += 1
            continue
        
        cnt_area = cv2.contourArea(cnt)
        rectangularity = cnt_area / float(rot_area) if rot_area > 0 else 0.0

        if rectangularity < RECT_MIN:
            if c_rect < debug_samples:
                print("[REJECT rect] rectangularity=%.2f (< %.2f)  cnt_area=%.0f  box_area=%.0f" %
                    (rectangularity, RECT_MIN, cnt_area, rot_area))
            c_rect += 1
            continue
  
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append((x, y, w, h))
        c_pass += 1
    
        if c_pass <= min(5, debug_samples):
            print("[PASS] x=%d y=%d w=%d h=%d  ar=%.2f rect=%.2f angle=%.1f area_frac=%.4f" %
                  (x, y, w, h, ar, rectangularity, angle, area_frac))

    print(f"[DEBUG] Filtered candidates: {len(candidates)}")
    print(f"[DEBUG] Skipped short: {c_short} | zero-dims: {c_zero} | angle: {c_angle}")
    print(f"[DEBUG] Rejected small: {c_small} | large: {c_large} | AR: {c_ar} | Rectangularity: {c_rect}")

    return candidates

# Function below sorts boxes by a measure of "goodness," which is defined by size and aspect ratio

def final_plates(candidates, img):
    
    H, W = img.shape[:2]
    best_box = None 
    best_score = 0

    for (x, y, w, h) in candidates:
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        distance = abs(aspect_ratio - 5.5)
        ar_score = max(0.0, min(1.0, 1 - distance / 1.0))
        area_norm = (w * h) / (W * H)
        combined_score = (ar_score + area_norm) / 2

        if combined_score > best_score:
            best_score = combined_score 
            best_box = (x, y, w, h)
    
    return best_box, best_score

contours = detect_candidates(img)
candidates = filter_plate_candidates(contours, img)
best_box, best_score = final_plates(candidates, img)

print("Best box:", best_box)
print("Best score:", round(best_score, 3))

vis = img.copy()
for (x,y,w,h) in candidates[:30]:
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.imshow("candidates", vis); cv2.waitKey(0); cv2.destroyAllWindows()


def read_plate_text(best_box, img):
    """
    Takes the best bounding box and image, crops the plate region,
    applies preprocessing and returns the detected text from the license plate.
    """

    (x, y, w, h) = best_box
    cropped_img = img[y:y+h, x:x+w]
    gray_plate = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    if gray_plate < 
    resize_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("resize_plate", resize_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

plate_text = read_plate_text(best_box, img)







    
