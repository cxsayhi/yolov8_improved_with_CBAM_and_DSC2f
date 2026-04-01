from ultralytics import YOLO
import cv2

# 1. 加载模型
model = YOLO('best_fixed.pt')

# 2. 执行推理
# stream=True 模式在处理视频或大量图片时更省显存
results = model.predict(source='000000234526.jpg', conf=0.4, save=True)

# 3. 处理结果
for r in results:
    print(r.boxes.cls)  # 打印检测到的类别 ID
    print(r.boxes.conf)  # 打印置信度
    print(r.boxes.xyxy)  # 打印坐标 (x1, y1, x2, y2)

    # 如果你想在 OpenCV 中查看结果
    # im_array = r.plot() # 绘制框图
    # cv2.imshow('Result', im_array)
    annotated_frame = results[0].plot()  # plot() 返回一个带框和标签的 NumPy 数组
    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(0)