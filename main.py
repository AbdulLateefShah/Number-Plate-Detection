import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from difflib import SequenceMatcher

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")

# Load Models
print("✅ Loading models...")
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

def normalize_plate(text):
    """Fix common OCR errors"""
    text = text.replace('O', '0').replace('I', '1').replace('Q', '0').replace('S', '5')
    return ''.join(c for c in text.upper() if c.isalnum())

def smart_reduce_duplicates(detections):
    """Smart duplicate reduction with detailed output"""
    def similarity(a, b):
        if a == b: return 1.0
        if abs(len(a) - len(b)) > 2: return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    groups = []
    for det in detections:
        plate = det['text']
        
        best_group = None
        best_sim = 0
        
        for group in groups:
            sim = similarity(plate, group[0]['text'])
            if sim > best_sim and sim >= 0.75:  # 75% similarity threshold
                best_sim = sim
                best_group = group
        
        if best_group:
            best_group.append(det)
        else:
            groups.append([det])
    
    print(f"\n🧠 SMART DUPLICATE REDUCTION:")
    print(f"{'='*60}")
    print(f"Input detections: {len(detections)}")
    print(f"Groups formed: {len(groups)}")
    
    unique = []
    merged_count = 0
    
    for i, group in enumerate(groups, 1):
        if len(group) > 1:
            plates = [d['text'] for d in group]
            confs = [f"{d['conf']:.3f}" for d in group]
            best = max(group, key=lambda x: x['conf'])
            
            print(f"Group {i:2d}: {plates}")
            print(f"         Confs: {confs}")
            print(f"         Best: {best['text']} (conf: {best['conf']:.3f})")
            print()
            
            unique.append(best)
            merged_count += len(group) - 1
        else:
            unique.append(group[0])
    
    print(f"📊 REDUCTION SUMMARY:")
    print(f"   Original unique count: {len(detections)}")
    print(f"   After smart reduction: {len(unique)}")
    print(f"   Duplicates merged: {merged_count}")
    print(f"   Improvement: {len(detections)} -> {len(unique)} truly unique plates")
    
    return unique

def preprocess_plate(image):
    """Enhanced preprocessing"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 40:
        gray = cv2.resize(gray, (int(w * 40/h), 40))
    
    versions = [
        cv2.adaptiveThreshold(cv2.bilateralFilter(gray, 11, 17, 17), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15),
        cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    return versions

def read_plate(plate_crop):
    """Enhanced OCR with normalization"""
    if plate_crop.size == 0:
        return "", 0.0
    
    best_text, best_conf = "", 0.0
    
    for processed in preprocess_plate(plate_crop):
        try:
            for bbox, text, conf in reader.readtext(processed, detail=1):
                clean_text = normalize_plate(text)
                if (4 <= len(clean_text) <= 8 and 
                    any(c.isdigit() for c in clean_text) and 
                    any(c.isalpha() for c in clean_text) and 
                    conf > best_conf):
                    best_text, best_conf = clean_text, conf
        except:
            continue
    
    return best_text, best_conf

# Video setup
input_video = "cars.mp4"
output_video = "cars_smart_fixed.mp4"
cap = cv2.VideoCapture(input_video)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print(f"🎬 Processing: {input_video} -> {output_video}")

# Processing
detections = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 5 == 0:
        if frame_count % 500 == 0:
            print(f"🔍 Progress: {frame_count/total_frames*100:.1f}% - Raw detections: {len(detections)}")
        
        results = model(frame, verbose=False)[0]
        
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            
            if conf > 0.2 and (x2-x1) > 20 and (y2-y1) > 10:
                plate_crop = frame[y1:y2, x1:x2]
                text, ocr_conf = read_plate(plate_crop)
                
                if text and ocr_conf > 0.5:
                    detections.append({'text': text, 'conf': ocr_conf})
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(text)*15+20, y1), (0, 255, 0), -1)
                    cv2.putText(frame, f"{text} ({ocr_conf:.2f})", (x1+5, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.putText(frame, f"Frame: {frame_count} | Detections: {len(detections)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)

cap.release()
out.release()

# SMART DUPLICATE REDUCTION PHASE
print(f"\n{'='*60}")
print("🧠 APPLYING SMART DUPLICATE REDUCTION")
print(f"{'='*60}")

# Group by basic text first (for comparison)
basic_groups = defaultdict(list)
for det in detections:
    basic_groups[det['text']].append(det)

basic_unique = [max(group, key=lambda x: x['conf']) for group in basic_groups.values()]

print(f"\n📊 BEFORE SMART REDUCTION:")
print(f"   Total detections: {len(detections)}")
print(f"   Basic unique count: {len(basic_unique)}")

# Apply smart reduction
smart_unique = smart_reduce_duplicates(basic_unique)

print(f"\n🏆 FINAL SMART RESULTS:")
print(f"📊 Total raw detections: {len(detections)}")
print(f"📊 Basic unique plates: {len(basic_unique)}")
print(f"📊 Smart unique plates: {len(smart_unique)}")
print(f"🎬 Video saved: {output_video}")

print(f"\n🥇 TOP TRULY UNIQUE PLATES:")
print("-" * 70)

smart_unique.sort(key=lambda x: x['conf'], reverse=True)

for i, det in enumerate(smart_unique[:20], 1):
    original_count = sum(1 for d in detections if d['text'] == det['text'])
    print(f"{i:2d}. {det['text']:8s} | "
          f"Conf: {det['conf']:.3f} | "
          f"Raw detections: {original_count:2d}")

# FIXED: Save results with UTF-8 encoding
try:
    with open('smart_final_results.txt', 'w', encoding='utf-8') as f:
        f.write("SMART ENHANCED ANPR RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input video: {input_video}\n")
        f.write(f"Total raw detections: {len(detections)}\n")
        f.write(f"Basic unique count: {len(basic_unique)}\n")
        f.write(f"Smart unique plates: {len(smart_unique)}\n")
        f.write(f"Improvement: {len(basic_unique)} -> {len(smart_unique)} truly unique\n\n")
        
        f.write("SMART UNIQUE PLATES (sorted by confidence):\n")
        f.write("-" * 50 + "\n")
        
        for i, det in enumerate(smart_unique, 1):
            original_count = sum(1 for d in detections if d['text'] == det['text'])
            f.write(f"{i:2d}. {det['text']} (conf: {det['conf']:.3f}) - {original_count} raw detections\n")

    print(f"\n💾 Detailed results saved to: smart_final_results.txt")
    print(f"🎯 SUCCESS: Reduced from {len(basic_unique)} to {len(smart_unique)} truly unique plates!")

except Exception as e:
    print(f"⚠️ File saving error (results still displayed above): {e}")
    print(f"🎯 SUCCESS: Reduced from {len(basic_unique)} to {len(smart_unique)} truly unique plates!")

print(f"\n🎉 ANPR SMART PROCESSING COMPLETE!")
print(f"📊 Final Result: {len(smart_unique)} truly unique vehicles detected!")
print(f"🎬 Annotated video: {output_video}")
print(f"📄 Results file: smart_final_results.txt")
