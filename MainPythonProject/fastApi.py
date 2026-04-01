from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI()

# 加载你训练好的模型
model = YOLO("best_fixed.pt")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 1. 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. 推理
    results = model(img)

    detections = []
    for r in results:
        # 获取归一化后的 xywh 坐标 (YOLO 格式)
        # xywhn: [x_center, y_center, width, height] 均为 0-1 之间的比例
        boxes = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            detections.append({
                "label": model.names[int(cls[i])],
                "confidence": float(conf[i]),
                "bbox": {
                    "x_center": float(boxes[i][0]),
                    "y_center": float(boxes[i][1]),
                    "width": float(boxes[i][2]),
                    "height": float(boxes[i][3])
                }
            })

    return {"count": len(detections), "results": detections}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)