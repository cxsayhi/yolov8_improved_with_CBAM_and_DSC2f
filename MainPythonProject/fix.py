from ultralytics import YOLO
import torch

# 正确的标签清单
correct_names = {
    0: 'bicycle', 1: 'car', 2: 'motorcycle', 3: 'airplane', 4: 'bus',
    5: 'train', 6: 'truck', 7: 'boat', 8: 'bird', 9: 'cat',
    10: 'dog', 11: 'horse', 12: 'sheep', 13: 'cow', 14: 'elephant',
    15: 'bear', 16: 'zebra', 17: 'giraffe', 18: 'banana', 19: 'apple',
    20: 'sandwich', 21: 'orange', 22: 'broccoli', 23: 'carrot',
    24: 'hot dog', 25: 'pizza', 26: 'donut', 27: 'cake'
}

# 2. 使用 YOLO 类加载模型（它会自动处理 weights_only 等安全问题）
model = YOLO('best.pt')

# 3. 修改模型内部的标签映射
# 修改预测器用的名称
model.model.names = correct_names
# 修改模型权重字典里的名称（为了永久保存）
if hasattr(model, 'ckpt') and model.ckpt:
    model.ckpt['names'] = correct_names

# 4. 保存模型
# 我们直接利用 torch.save 保存 model.ckpt，这是最完整的保存方式
torch.save(model.ckpt, 'best_fixed.pt')

print("✅ 修复完成！新文件为 'best_fixed.pt'")
print("验证一下：", YOLO('best_fixed.pt').names[1]) # 应该输出 'car'