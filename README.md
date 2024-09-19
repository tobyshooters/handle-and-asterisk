![handle-and-asterisk](./handle-and-asterisk.jpg)

```
from model import Clip
m = Clip()
m.embed_text("a handle and an asterisk are just three asterisks")
```

download weights:
[ViT-B/32 image model](https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_image_model_vitb32.onnx)
& [ViT-B/32 text model](https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_text_model_vitb32.onnx)

a pruned version of https://github.com/lakeraai/onnx_clip
