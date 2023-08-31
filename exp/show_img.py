import torch as th
from pathlib import Path
import matplotlib.pyplot as plt

x_0 = th.load(Path.cwd() / "outputs/x_0.pth")[0, 0, :, :]  # .detach().numpy()
plt.imshow(x_0)
plt.show()
