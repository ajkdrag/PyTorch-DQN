import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(exploration, score, out_dir):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(exploration, label="exploration", color="g")
    ax2.plot(score, label="score", color="r")

    ax1.set_xlabel("run")
    ax1.set_ylabel("exploration", color="g")
    ax2.set_ylabel("score", color="r")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "results.png", bbox_inches="tight")
