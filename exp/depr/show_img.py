import torch as th
from pathlib import Path
import matplotlib.pyplot as plt

x_0 = th.load(Path.cwd() / "outputs/unet_samples.th")[0, 0, :, :].detach().cpu().numpy()
plt.imshow(x_0)
plt.show()
