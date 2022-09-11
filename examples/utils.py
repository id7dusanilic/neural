import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuring default matplotlib parameters
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 12
mpl.rcParams["image.cmap"] = "gray"


imgToFloat = lambda img: img.astype(np.float32) / np.iinfo(img.dtype).max


imgToUint8 = lambda img: (img * np.iinfo(np.uint8).max).astype(np.uint8)


def showImage(img, fig=None, axis=None, figsize=None, show=True, imshowKwords={}, **kwords):
    toShow = img

    fig, axis = plt.subplots() if fig is None else (fig, axis)

    if figsize is not None:
        oldFigsize = mpl.rcParams["figure.figsize"]
        mpl.rcParams["figure.figsize"] = figsize

    fig.tight_layout()

    axis.imshow(toShow, **imshowKwords)
    axis.axis("off")

    if "title" in kwords:
        axis.set_title(kwords["title"])

    if show:
        plt.show()

    if figsize is not None:
        mpl.rcParams["figure.figsize"] = oldFigsize

    if not show:
        return fig, axis


def normalize(tensor, mean, std):
    new = tensor / np.max(np.abs(tensor))
    return (new - mean) / std


def showMNIST(img, ps):
    fig, (ax1, ax2) = plt.subplots(figsize=(12,8), ncols=2)

    showImage(img, axis=ax1, fig=fig, show=False)

    ax2.barh(np.arange(ps.size), ps.squeeze(), height=0.6, tick_label=np.arange(ps.size))

    for s in ['top', 'bottom', 'left', 'right']:
        ax2.spines[s].set_visible(False)

    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')

    ax2.xaxis.set_tick_params(pad=5)
    ax2.yaxis.set_tick_params(pad=10)

    ax2.set_xlim([0.0, 1.0])
    ax2.set_title("Model performance", fontsize=20)

    ax2.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

def plotLossTrack(losses, labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    for loss, label in zip(losses, labels):
        ax.plot(loss.flatten(), label=label)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration number")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid("both")
