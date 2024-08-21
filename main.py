from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('best.pt')

def object(img):
    results = model.predict(img)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    object_classes = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        conf = float(row[4])
        d = int(row[5])
        obj_class = "orange"  # class_list[d]
        
        confidence_text = f"{obj_class}: {conf:.2f}"
        
        bounding_box_color = (0, 0, 255)  # Red color for the bounding box
        text_color = (255, 255, 255)  # White color for text
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2
        text_offset_x = 5
        text_offset_y = 5

        cv2.rectangle(img, (x1, y1), (x2, y2), bounding_box_color, 2)

        (text_width, text_height) = cv2.getTextSize(confidence_text, text_font, text_size, text_thickness)[0]
        text_background_width = text_width + 2 * text_offset_x
        text_background_height = text_height + 2 * text_offset_y

        cv2.rectangle(img, (x1, y1 - text_background_height), (x1 + text_background_width, y1), bounding_box_color, -1)

        text_x = x1 + text_offset_x
        text_y = y1 - text_offset_y
        cv2.putText(img, confidence_text, (text_x, text_y), text_font, text_size, text_color, text_thickness)
        object_classes.append(obj_class)
    return object_classes

def count_objects_in_all_images(images):
    total_orange_count = 0
    object_counts = {}
    for img in images:
        object_classes = object(img.copy())
        orange_count = object_classes.count("orange")
        total_orange_count += orange_count
        
        for obj, count in Counter(object_classes).items():
            object_counts[obj] = object_counts.get(obj, 0) + count

    print("Total Orange Count in All Images:", total_orange_count)
    print("Object Counts in All Images:", object_counts)
    
    return total_orange_count, object_counts

@app.post('/orange-counting')
async def orange_detection(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    image4: UploadFile = File(...),
):
    images = []
    files = [image1, image2, image3, image4]
    
    for file in files:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 640))
        images.append(img)

    if len(images) == 4:
        total_orange_count, _ = count_objects_in_all_images(images)
        return JSONResponse(content={"total_orange_count": total_orange_count})
    else:
        raise HTTPException(status_code=400, detail="Not all images were loaded properly")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)
