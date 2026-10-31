"""TEED 가중치를 teed.onnx 로 변환하는 스크립트.

사용법:
  git clone https://github.com/xavysp/TEED
  cd TEED
  pip install torch onnx
  python export_teed.py [체크포인트.pth]

체크포인트를 안 주면 checkpoints/ 아래에서 .pth 파일을 자동으로 찾아요.
(Hugging Face의 7_model.pth 같은 파일도 경로를 직접 넘기면 돼요.)
만들어진 teed.onnx 를 index.html 과 같은 폴더에 두면 됩니다.

입력 : [1, 3, H, W] float32, RGB, 값 범위 0~255 (H, W는 16의 배수)
출력 : [1, 1, H, W] 엣지 확률 (모든 출력의 로짓 평균 -> sigmoid)
"""
import glob
import sys

import torch

from ted import TED  # TEED 저장소의 ted.py


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        outs = self.m(x)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        return torch.sigmoid(torch.stack(list(outs), 0).mean(0))


def find_ckpt():
    if len(sys.argv) > 1:
        return sys.argv[1]
    found = sorted(glob.glob("checkpoints/**/*.pth", recursive=True))
    if not found:
        sys.exit("체크포인트(.pth)를 찾지 못했어요. 경로를 인자로 넘겨 주세요.")
    return found[-1]


ckpt = find_ckpt()
print("checkpoint:", ckpt)

model = TED()
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

dummy = torch.zeros(1, 3, 512, 512)
torch.onnx.export(
    Wrap(model),
    dummy,
    "teed.onnx",
    input_names=["image"],
    output_names=["edge"],
    dynamic_axes={"image": {2: "h", 3: "w"}, "edge": {2: "h", 3: "w"}},
    opset_version=17,
)
print("saved teed.onnx")
