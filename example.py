import ecg_plot
import argparse
import matplotlib.pyplot as plt
import preprocess
import os
import read_ecg
import numpy as np
import torch
import torch.nn as nn
from generate_heatmap import GradCAM, heatmap

# CHANGED: prefer local modules over any similarly named installed packages
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CHANGED: silence PyTorch’s future-warning from backward hooks
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ModelBaseline(nn.Module):
    def __init__(self ,):
        super(ModelBaseline, self).__init__()
        self.kernel_size = 17

        # conv layer
        downsample = self._downsample(4096, 1024)
        self.conv1 = nn.Conv1d(in_channels=8,
                               out_channels=16,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        downsample = self._downsample(1024, 256)
        self.conv2 = nn.Conv1d(in_channels=16,
                               out_channels=32,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        downsample = self._downsample(256, 32)
        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=64,
                               kernel_size=self.kernel_size,
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)

        # linear layer
        self.lin = nn.Linear(in_features=32 * 64,
                             out_features=1)

        # ReLU
        self.relu = nn.ReLU()

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x_flat= x.view (x.size(0), -1)
        x = self.lin(x_flat)

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ECG from wfdb')

    parser.add_argument('path_to_ecg', type=str,
                        help='Path to the file to be plot.')
    parser.add_argument('path_to_model', type=str,
                        help='Path to model weights.')
    parser.add_argument('--save', default="",
                        help='Save in the provided path. Otherwise just display image.')
    parser = preprocess.arg_parse_option(parser)
    parser = read_ecg.arg_parse_option(parser)
    args = parser.parse_args()
    print(args)

    row_height = 4
    cols = 1

    ecg, sample_rate, leads = read_ecg.read_ecg(args.path_to_ecg, format=args.fmt)

    # CHANGED: remove unsupported 'powerline' kwarg
    ecg, sample_rate, leads = preprocess.preprocess_ecg(ecg, sample_rate, leads,
                                                        new_freq=400,
                                                        new_len=4096,
                                                        scale=args.scale,
                                                        use_all_leads=False,
                                                        remove_baseline=True)

    model = ModelBaseline()
    ckpt = torch.load(args.path_to_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    x = torch.Tensor(ecg)[None, :, :]
    y_predicted = model(x)

    probs = torch.sigmoid(y_predicted).cpu().detach().numpy().flatten()
    print('Prob(1) = {:0.3f}'.format(*probs))

    grad_cam_model = GradCAM(model)
    _ = grad_cam_model(x)  # not using the result, but need it for initialisation

    # CHANGED: use deeper layer + normalize/broadcast CAM so it’s clearly visible
    x_viz = grad_cam_model.generate('conv3')
    x_viz = x_viz.detach().cpu().data.numpy()
    if x_viz.ndim == 3:
        x_viz = x_viz.squeeze(0)
    if x_viz.shape[0] != ecg.shape[0]:
        x_viz = np.tile(x_viz.mean(axis=0, keepdims=True), (ecg.shape[0], 1))
    x_viz = np.maximum(x_viz, 0)
    x_viz = (x_viz - x_viz.min(axis=1, keepdims=True)) / (x_viz.max(axis=1, keepdims=True) - x_viz.min(axis=1, keepdims=True) + 1e-8)

    ecg_plot.plot(ecg, sample_rate=sample_rate,
                  lead_index=leads, style='bw',
                  row_height=row_height, columns=cols)
    # CHANGED: stronger overlay scale so you can see it
    heatmap(ecg, x_viz, sample_rate=sample_rate, columns=cols, scale=10, row_height=row_height)

    # rm ticks
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)

    # CHANGED: save the current Matplotlib figure so the overlay is included
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
    else:
        plt.show()
