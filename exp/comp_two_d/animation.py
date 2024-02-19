"""Generate animation of 2D comp sampling trajectories"""
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

trajs_rev = th.load(Path.cwd() / "results/comp_two_d/trajs_rev.th")
trajs_hmc = th.load(Path.cwd() / "results/comp_two_d/trajs_hmc_with_rev.th")


T = trajs_rev.size(0) - 1
slow = 10
t_start = 91

fig, (ax_rev, ax_hmc) = plt.subplots(1, 2, figsize=(12, 6))
(rev_h,) = ax_rev.plot(trajs_rev[t_start, :, 0], trajs_rev[0, :, 1], ".")
ax_rev.set_title("Reverse")
(hmc_h,) = ax_hmc.plot(trajs_hmc[t_start, :, 0], trajs_hmc[0, :, 1], ".")
ax_hmc.set_title("HMC")

frame_text = fig.text(0.5, 0.05, f"t={T}", ha="center", va="center")


def update(frame):
    t = frame + t_start
    if t < slow or t > T - slow:
        ani._interval = 500
    else:
        ani._interval = 30
    rev_h.set_xdata(trajs_rev[t, :, 0])
    rev_h.set_ydata(trajs_rev[t, :, 1])

    hmc_h.set_xdata(trajs_hmc[t, :, 0])
    hmc_h.set_ydata(trajs_hmc[t, :, 1])

    frame_text.set_text(f"t={T - t}")

    return rev_h, hmc_h, frame_text


def func(current_frame: int, total_frames: int):
    print(current_frame / total_frames)


ani = animation.FuncAnimation(fig=fig, func=update, frames=slow)

writergif = animation.PillowWriter(fps=2)
ani.save("anim_t_9_to_0.gif", writer=writergif)
plt.show()
