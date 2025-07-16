import os
import torch
import yaml
from models.lmscnet import LMSCNetModel
import argparse
import numpy as np

class JitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = {'3D_OCCUPANCY': input_tensor}
        out = self.model(x) # Output logits

        logits = out['pred']
        return torch.argmax(logits, dim=1) # Output: [B, W, H, D] labels
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config YAML used during training')
    parser.add_argument('--output', type=str, default='lmscnet_scripted.pt', help='Output JIT model path')
    args = parser.parse_args()

    # Load config
    with open(args.cfg, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init model
    model = LMSCNetModel(cfg_dict)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    # Wrap for JIT
    wrapped = JitWrapper(model)

    # Dummy input
    vmin = np.array(cfg_dict['data']['volume_size_min'])
    vmax = np.array(cfg_dict['data']['volume_size_max'])
    voxel_size = cfg_dict['data']['voxel_size']
    input_dims = np.round((vmax - vmin) / voxel_size).astype(int)    

    shape = input_dims
    dummy_input = torch.zeros((1, 1, shape[0], shape[1], shape[2]), dtype=torch.float32).to(device)
    print("Gonna script and save")

    # Script and save
    scripted = torch.jit.trace(wrapped, dummy_input)
    scripted.save(args.output)
    print(f"Scripted model saved to {args.output}")

if __name__ == "__main__":
    main()