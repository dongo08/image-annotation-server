# Image Annotation API

在图片上绘制白色矩形标注框并插入居中文本，支持自动换行、自适应字号、Emoji。

## 安装

```bash
pip install -r requirements.txt
```

需要 Windows 系统字体（微软雅黑 / 黑体 / 宋体 + Segoe UI Emoji）。

## 启动

```bash
python server.py
# → http://0.0.0.0:8700  (IPv4)
# → http://[::]:8700      (IPv6)
```

## API

### `POST /annotate`

`multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | file | 原始图片（jpg/png 等） |
| `annotations` | string | JSON 数组字符串 |

### annotations 格式

```json
[
  {
    "x": 0.1,
    "y": 0.1,
    "width": 0.3,
    "height": 0.12,
    "text": "标注文本"
  }
]
```

| 参数 | 范围 | 说明 |
|------|------|------|
| `x`, `y` | 0–1 | 矩形左上角位置，0=左/上，1=右/下 |
| `width`, `height` | 0–1 | 矩形宽高，相对图片尺寸的比例 |
| `text` | string | 矩形内显示的文本 |

### 示例（Python）

```python
import requests
import json

with open("photo.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8700/annotate",
        files={"image": f},
        data={"annotations": json.dumps([
            {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.1, "text": "Hello 😀"}
        ])}
    )

with open("output.png", "wb") as f:
    f.write(resp.content)
```

