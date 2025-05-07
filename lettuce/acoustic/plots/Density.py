import matplotlib.pyplot as plt
import numpy as np
from abc import ABC

class Plot(ABC):
    def __init__(self,
                 base: str = None,
                 show: bool = True,
                 style: str =  "/home/mbedru3s/Dokumente/cluster_hbrs_home/plopy/plopy/styles/ecostyle.mplstyle"):
        self.show = show
        self.base = base
        if style:
            plt.style.use(style)

    def __call__(self):
        fig, ax = plt.subplots()
        self.setup_plot(ax)
        self.plot_graphs(ax)
        self.plot_annotations(ax)

        self.add_legend(ax)
        self.standard_export(name="Density_L2", png=False, pdf=True)

        # plt.ylim(0., 0.000175)
        # # plt.yscale("log")
        # plt.show()

    def setup_plot(self, ax):
        ax.grid(visible=True, which='major', axis='y')
        ax.tick_params(axis="y", direction="in", pad=0)
        ax.set_title(r"\noindent\footnotesize{L$^2$}", ha='right')
        ax.set_title(r"\noindent\textbf{Density Error} \textendash{} \footnotesize{Convected Vortex}", loc='left')
        ax.set_xlabel(r"\textit{Time}", style='italic', color='#525254')
        ax.set_ylim([0, 0.0145])
        plt.yscale("log")
        ylabels_idx = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ylabels_str = [r"$1e^{-6}$", r"$1e^{-5}$", r"$1e^{-4}$", r"$1e^{-3}$", r"$1e^{-2}$"]
        # ylabels_str = []
        ax.set_yticks(ylabels_idx)
        ax.set_yticklabels(ylabels_str, ha='right')
        ax.axis([-0.2,6.2,1e-7,10e-2])

    def add_legend(self, ax):
        ax.plot([0, 1.75],2*[1e-2],color="white",)
        ax.plot([0, 1.75],2*[1e-3],color="white",)
        handles, labels = ax.get_legend_handles_labels()
        order = np.arange(len(handles))

        legend = ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                            loc=2, bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=1, columnspacing=.5, fontsize=6);

        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        frame.set_alpha(1)
    def plot_graphs(self, ax):
        result_k10_k20 = np.load('result_k10_k20.npy')
        result_k10_k21 = np.load('result_k10_k21.npy')
        result_k10_k25 = np.load('result_k10_k25.npy')
        result_k11_k20 = np.load('result_k11_k20.npy')
        result_k11_k21 = np.load('result_k11_k21.npy')
        result_k11_k22 = np.load('result_k11_k22.npy')
        result_k11_k23 = np.load('result_k11_k23.npy')
        result_k10_k22 = np.load('result_k10_k22.npy')
        result_k10_k23 = np.load('result_k10_k23.npy')
        result_n1 = np.load('result_model_training_v1_18_553854.npy')
        result_n2 = np.load('result_model_training_v1_18_553848.npy')
        result_n3 = np.load('result_model_training_v1_17.npy')



        ax.plot(result_k10_k20[0], result_k10_k20[1], color="black", marker='', linestyle=':', linewidth=1, label=r"$\sigma=0$, $K_2=0$")
        ax.plot(result_k10_k21[0], result_k10_k21[1], color="black", marker='', linestyle='-', linewidth=1, label=r"$\sigma=0$, $K_2=0$")
        ax.plot(result_k11_k20[0], result_k11_k20[1], color="black", marker='', linestyle='--', linewidth=1, label=r"$\sigma=1$, $K_2=0$")
        ax.plot(result_k10_k25[0], result_k10_k25[1], color="black", marker='', linestyle='-.', linewidth=1, label=r"$\sigma=0$, $K_2=5$")
        # ax.plot(result_k11_k21[0], result_k11_k21[1], color="black", marker='', linestyle='-', linewidth=1, label="K1=1, K2=1")
        ax.plot(result_k11_k22[0], result_k11_k22[1], color="black", marker='', linestyle=(1, (2, 6)), linewidth=1, label=r"$\sigma=1$, $K_2=2$")
        # ax.plot(result_k11_k23[0], result_k11_k23[1], color="green", marker='', linestyle='-', linewidth=1, label="K1=1, K2=3")
        ax.plot(result_k10_k22[0], result_k10_k22[1], color="black", marker='', linestyle=(5, (10, 3)), linewidth=1, label=r"$\sigma=0$, $K_2=2$")
        # ax.plot(result_k10_k23[0], result_k10_k23[1], color="black", marker='', linestyle='-', linewidth=1, label="K1=0, K2=3")

        # ax.plot(result_n1[0], result_n1[1], color="red", marker='', linestyle="-", linewidth=1, label="Neural Network")
        # ax.plot(result_n2[0], result_n2[1], color="blue", marker='', linestyle=(5, (10, 3)), linewidth=1, label="K1=0, K2=2")
        ax.plot(result_n3[0], result_n3[1], color="red", marker='', linestyle="-", linewidth=1.5, label="Neural Network")

    def plot_annotations(self, ax):
        ax.annotate("Pressure Wave", xy=(0.55,3e-6),xytext=(0.05,3.5e-5),
                    arrowprops=dict(arrowstyle= '-|>',
                                    color='black',
                                    lw=.5,
                                    ls='--')
                    )
        ax.annotate("Convected Vortex", xy=(1.6,3e-7),xytext=(2.5,2e-7),
                    arrowprops=dict(arrowstyle= '-|>',
                                    color='black',
                                    lw=.5,
                                    ls='--')
                    )
    def standard_export(self,
                        name: str = "output",
                        png: bool = True,
                        pdf: bool = True):
        if self.base:
            if png:
                plt.savefig(self.base + name + ".png",
                            format='png',
                            bbox_inches='tight',
                            pad_inches=0.01,
                            dpi=300,
                            transparent=False)
            if pdf:
                plt.savefig(self.base + name + ".pdf",
                            format='pdf',
                            bbox_inches='tight',
                            pad_inches=0.01,
                            transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()

plot = Plot(base="./")
plot()



# plt.plot(result[0], result[1], marker='', linestyle='-',label="current",color="black")

