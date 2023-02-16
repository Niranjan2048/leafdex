import torch
from torch_model import Model # Made up package

device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')

model = Model()
model.load_state_dict(torch.load('leafsnap_model.pth'))

model = model.to(device) # Set model to gpu
model.eval()

inputs = torch.random.randn(1, 3, 224, 224) # Dtype is fp32
inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu

# Run forward pass
with torch.no_grad():
  pred = model(inputs)

# Do something with pred
pred = pred.detach().cpu().numpy() # remove from computational graph to cpu and as numpy