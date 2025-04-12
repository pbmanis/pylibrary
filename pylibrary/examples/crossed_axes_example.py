import matplotlib.pyplot as mpl
import numpy as np
import mpl_toolkits.axisartist.axislines as AL

def test():
    fig = mpl.figure(1)
    ax = AL.AxesZero(fig, 111)
    fig.add_subplot(ax)

    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

    x = np.linspace(-0.5, 1., 100)
    ax.plot(x, np.sin(x*np.pi))

    mpl.show()

if __name__ == "__main__":
    test()
