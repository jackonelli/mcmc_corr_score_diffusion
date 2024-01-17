import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

trajs_rev = th.load(Path.cwd() / "results/comp_two_d/trajs_rev.th")
trajs_hmc = th.load(Path.cwd() / "results/comp_two_d/trajs_hmc_with_rev.th")

fig, (ax_rev, ax_hmc) = plt.subplots(1, 2, figsize=(12, 6))
(rev_h,) = ax_rev.plot(trajs_rev[0, :, 0], trajs_rev[0, :, 1], ".")
ax_rev.set_title("Reverse")
(hmc_h,) = ax_hmc.plot(trajs_hmc[0, :, 0], trajs_hmc[0, :, 1], ".")
ax_hmc.set_title("HMC")

frame_text = fig.text(0.5, 0.05, "", ha="center", va="center")

T = trajs_rev.size(0) - 1
slow = 7


def update(frame):
    if frame < slow or frame > T - slow:
        ani._interval = 500
    else:
        ani._interval = 30
    rev_h.set_xdata(trajs_rev[frame, :, 0])
    rev_h.set_ydata(trajs_rev[frame, :, 1])

    hmc_h.set_xdata(trajs_hmc[frame, :, 0])
    hmc_h.set_ydata(trajs_hmc[frame, :, 1])

    frame_text.set_text(f"t={T - frame}")

    return rev_h, hmc_h, frame_text


def func(current_frame: int, total_frames: int):
    print(current_frame / total_frames)


ani = animation.FuncAnimation(fig=fig, func=update, frames=T + 1)
writergif = animation.PillowWriter(fps=30)

# ani.save("anim.gif", writer=writergif)
plt.show()
