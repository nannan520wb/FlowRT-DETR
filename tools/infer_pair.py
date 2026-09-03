import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
from src.data.transforms import ComputeOpticalFlow

class ModelWrapper(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, prev_images, flow, orig_target_sizes):
        outputs = self.model(images, x_prev=prev_images, flow=flow)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def draw_pil(image, labels, boxes, scores, thrh=0.5, save_path="result.jpg"):
    """Draw filtered detections on a PIL image."""
    draw = ImageDraw.Draw(image)
    valid_idx = scores > thrh
    valid_labels = labels[valid_idx]
    valid_boxes = boxes[valid_idx]
    valid_scores = scores[valid_idx]

    for i, b in enumerate(valid_boxes):
        draw.rectangle(list(b), outline='red', width=2)
        text = f"cls: {valid_labels[i]} {valid_scores[i]:.2f}"
        draw.text((b[0], max(0, b[1] - 10)), text=text, fill='blue')

    image.save(save_path)
    print(f"Saved {len(valid_boxes)} detections to: {save_path}")


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']

    cfg.model.load_state_dict(state)
    model = ModelWrapper(cfg).to(args.device)
    model.eval()

    flow_computer = ComputeOpticalFlow(
        ckpt_path=args.flow_ckpt,
        config_path=args.flow_config,
        device=args.device,
    )

    print(f"Current frame: {args.im_file}")
    print(f"Previous frame: {args.prev_im_file}")
    curr_pil_img = Image.open(args.im_file).convert('RGB')
    prev_pil_img = Image.open(args.prev_im_file).convert('RGB')

    w, h = curr_pil_img.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    with torch.no_grad():
        flow_img = flow_computer(curr_pil_img, prev_pil_img)

        pad_h = max(0, 544 - h)
        pad_w = max(0, 960 - w)

        normalize_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        curr_data = normalize_transform(F.pad(curr_pil_img, (0, 0, pad_w, pad_h), fill=0))[None].to(args.device)
        prev_data = normalize_transform(F.pad(prev_pil_img, (0, 0, pad_w, pad_h), fill=0))[None].to(args.device)

        flow_data_tensor = torch.as_tensor(flow_img).clone().detach()

        flow_data = F.pad(flow_data_tensor, (0, 0, pad_w, pad_h), fill=0)[None].to(args.device)

        output = model(curr_data, prev_data, flow_data, orig_size)
        labels, boxes, scores = output

        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

    draw_pil(curr_pil_img, labels, boxes, scores, thrh=args.thresh, save_path=args.out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlowRT-DETR inference on an image pair.')
    parser.add_argument('-c', '--config', required=True, help='Model configuration.')
    parser.add_argument('-r', '--resume', required=True, help='Detector checkpoint.')
    parser.add_argument('-f', '--im-file', required=True, help='Current frame.')
    parser.add_argument('-pf', '--prev-im-file', required=True, help='Earlier frame.')
    parser.add_argument('--flow-config', default='optical_flow/config/kitti-S.json')
    parser.add_argument('--flow-ckpt', default='optical_flow/weights/kitti-S.pth')
    parser.add_argument('-o', '--out-file', default='infer_result.jpg')
    parser.add_argument('-t', '--thresh', type=float, default=0.5)
    parser.add_argument('-d', '--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    main(args)
