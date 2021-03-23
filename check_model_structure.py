import torch
import torchvision.models as models
from torchsummary import summary

checkpoint = './checkpoints/SSD_940_9809.pth'
checkpoint = torch.load(checkpoint)  # , map_location=torch.device('cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = checkpoint['model']
model = model.to(device)
ssd = model.to(device)

summary(ssd, (3, 300, 300))
