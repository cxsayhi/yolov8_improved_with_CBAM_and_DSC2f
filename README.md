## 项目简介：

本项目给传统的yolov8添加cbam模块，以及深度可分离的c2f模块，实现了模型的改进和微调。（最后效果未达预期）

## contents:

`/MainPythonProject `:加载并运行模型的文件夹

- /MainPythonProject/fix.py 由于train模块当中模型label和实际序号发生了错位，所以当前文件夹下的best.pt需要通过该代码处理，得到正确的权重best_fixed.pt。（现data.yaml当中的顺序已修正)
- /MainPythonProject/run.py: 模型运行代码
- /MainPythonProject/fastApi.py：向java提供服务

`/trainig`:模型修改后的结构和训练的文件夹。

- `/train/train.ipynb`:训练代码
- `/train/Coco_dataset_download.ipynb` : Coco数据集下载代码
- `/train/data.yaml `: 数据集定义和配置文件
- `/train/yolov8n_cbam_dsconv` :模型结构文件



## 主项目运行方式：

打开MainPythonProject，修改：

- 选择模型
- 检测对象

即可运行。

模型的训练和修改在文件夹ultralytics内，训练方式如下：

## yolov8 colab训练和模块改进方式：

### 1.连接googledrive

```
from google.colab import drive
drive.mount('/content/drive')
```

### 2.安装ultralytics依赖

```
!git clone https://github.com/ultralytics/ultralytics.git
!pip install ultralytics
```

#### 强行让 Python 加载你的修改版源码

在 Colab 代码的最顶端（导入 YOLO 之前），将你的源码路径插入到系统路径的最前面：

```
import sys
import os

# 将你修改过的那个 ultralytics 文件夹的父目录加入系统路径
# 注意：路径要指向包含 ultralytics 文件夹的那个目录
sys.path.insert(0, '/content/ultralytics')

from ultralytics import YOLO
# 验证路径：运行下面这行，确保输出的是 /content/ultralytics/...
import ultralytics
print(ultralytics.__file__)
```



### 3.切换到文件目录

```
%cd /content/drive/MyDrive/Colab Notebooks/yolo_detection/yolov8_tuning/yolov8_fine_tuning
```

### 4.修改ultralytics文件夹中的文件，添加定义的cbam模块和C2f_DS模块，和相关使用到的模块

#### 在 /content/ultralytics/ultralytics/nn/modules/__init__.py中

#### **将原本的导入添加C2f_DS, DSBottleneck, DSConv,：**

```
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    C2f_DS,


)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
    DSBottleneck,
    DSConv,

)
```

#### 在\__all__当中添加"C2f_DS","DSConv","DSBottleneck",

```
__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "OBB",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "Focus",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "C2f_DS",
    "DSConv",
    "DSBottleneck",
)

```

#### 在/content/ultralytics/ultralytics/nn/modules/block.py当中

#### 将\__all__当中添加 "C2f_DS"

```
__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "C2f_DS",

)
```

#### 添加C2f_DS模块

```
from .conv import DSBottleneck

class C2f_DS(C2f): # Now Python knows what 'C2f' is
    """A C2f block that uses DSBottlenecks."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Override the 'm' (bottlenecks module list) with our custom DSBottleneck
        self.m = nn.ModuleList(DSBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
```

#### 在/content/ultralytics/ultralytics/nn/modules/conv.py当中

#### 在\__all__当中添加DSConv，DSBottleneck的声明：

```
__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "DSConv",
     "DSBottleneck",
)
```

#### 将ChannelAttention，SpatialAttention和CBAM的代码修改为：

```
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):  # c1: in_channels, c2: out_channels
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
```

#### 添加DSConv的代码和DSBottleneck的代码

```
import torch.nn as nn
# Add import for Mish
from torch.nn import Mish

# ... (在 conv.py 中已有的其他类定义之后) ...

class DSConv(nn.Module):
    """
    Depthwise Separable Convolution module.
    Replaces a standard Conv module to reduce parameters and computation.
    """
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        # Depthwise Conv: Groups equal to input channels to perform
        # separate convolution for each channel.
        self.depthwise = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        # Pointwise Conv: Uses 1x1 convolution to change the number of channels.
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        # Activation function
        # 将 SiLU 替换为 Mish
        self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # Apply depthwise convolution
        x = self.act(self.bn1(self.depthwise(x)))
        # Apply pointwise convolution
        x = self.act(self.bn2(self.pointwise(x)))
        return x

class DSBottleneck(nn.Module):
    """A Bottleneck block using DSConv for lighter weight."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DSConv(c1, c_, k=3, s=1)
        self.cv2 = DSConv(c_, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

#### 在/content/ultralytics/ultralytics/nn/task.py当中导入CBAM, C2f_DS, DSConv, DSBottleneck,

```
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    CBAM,
    C2f_DS,    
    DSConv,
    DSBottleneck,
)
```

#### 添加声明CBAM,和C2f_DS,

```
base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            CBAM,
            C2f_DS,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
            CBAM,
            C2f_DS,
        }
    )
```



### 5.重启对话后运行代码

```
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow # Colab 专用显示函数
import os


model = YOLO('/content/drive/MyDrive/Colab Notebooks/yolo_detection/yolo_revised_2/yolov8n_cbam_dsconv.yaml')

    # ✅ 加入以下代码，打印出模型实际加载的 YAML 内容
    # print("\n" + "="*25 + " 模型实际加载的 YAML 内容 " + "="*25)
    # print(yaml.dump(model.yaml))
    # print("="*70 + "\n")

    # 2. Load the pre-trained weights (e.g., yolov8n.pt)
    # The .load() method intelligently transfers matching layers from the .pt file
model.load('yolov8n.pt')

    # 3. Start training
model.train(
        data='/content/drive/MyDrive/Colab Notebooks/yolo_detection/yolov8_object_detection/dataset/data/data.yaml',
        pretrained='yolov8n.pt',
        epochs=200,
        imgsz=640,
        plots=True,
        patience=10,
        batch=64,
        #device='cuda'
    )  
```



PS：点击页面保持colab页面活跃

```
function ConnectButton() {
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton, 60000); // 每 60 秒点击一次
```





