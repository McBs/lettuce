import plopy
import torch
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PlotNeuralNetwork",
]

class PlotNeuralNetwork(plopy.Plot):

    def loss_function(self, loss=None, epochs=None, name="loss_function"):
        fig, ax1 = plt.subplots()
        ax1.grid(visible=True, which='major', axis='y')
        ax1.tick_params(axis="y", direction="in", pad=0)
        ax1.set_title(r"\noindent\footnotesize{$L$}", ha='right')
        ax1.set_title(r"\noindent\textbf{Loss} \textendash{} \footnotesize{TGV3D}", loc='left', )
        ax1.set_title("L", ha='right')
        ax1.set_title("Loss", loc='left', )
        ax1.set_xlabel("Epochs", style='italic', color='#525254')

        DarkGray = "#222222"
        epochs = np.arange(1,len(loss)+1) if epochs is None else epochs
        plt.plot(epochs, loss, linewidth=1.5, color=DarkGray, label=r'Loss')

        self.standard_export(name=name,
                             png=False,
                             pdf=True)
        export = False
        if export:
            return fig, ax1



if __name__ == "__main__":
    x = torch.arange(10)
    y = torch.exp(x)

    plot = PlotNeuralNetwork(
        base="./",
        style="/home/mbedru3s/science/plopy/plopy/styles/ecostyle.mplstyle")
    plot.loss_function(y)
