import torch
import cv2
from torchvision import transforms
from utils.MiniFASNet import MiniFASNetV2  
from utils.utility import get_kernel, parse_model_name
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.path.basename("models/2.7_80x80_MiniFASNetV2.pth")
h_input, w_input, model_type, _ = parse_model_name(model_name)
kernel_size = get_kernel(h_input, w_input)
model = MiniFASNetV2(conv6_kernel=kernel_size).to(DEVICE)
state_dict=torch.load("models/2.7_80x80_MiniFASNetV2.pth", map_location=DEVICE)
model.load_state_dict(state_dict,strict=False)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def is_live(face_img):
    face = cv2.resize(face_img, (80, 80))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(face)
        score = torch.softmax(out, dim=1)[0][1].item()

    return score > 0.7