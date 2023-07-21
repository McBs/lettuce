import numpy as np
import torch
import matplotlib.pyplot as plt

__all__ = [
    "Plot"
]
class Plot:
    def __init__(self, filebase=None, show=True, style='/home/mbedru3s/Dokumente/lettuce/lettuce/plots/ecostyle.mplstyle'):
        self.show = show
        self.filebase = filebase
        plt.style.use(style)


    def __call__(self, f):
        raise NotImplementedError

    def basic(self, title='chart', xlabel='x', log=False, axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        plt.title(r"\noindent\textbf{" + title + "}", x=1, y=1.075)
        plt.xlabel(r"\textit{" + xlabel + "}")
        if log:
            plt.xscale('log')
            plt.yscale('log')

        if axis:
            plt.axis(axis)

        for key, value in kwargs.items():
            if postprocess:
                value = postprocess(value)
            if isinstance(value, list):
                plt.plot(*value, linestyle='-', label=key)
            else:
                plt.plot(value, linestyle='-', label=key)

        handles, labels = ax1.get_legend_handles_labels()
        order = np.arange(len(handles))
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc=2, bbox_to_anchor=(-0.02, 1.25), frameon=False, ncol=4, columnspacing=1, fontsize=8)

        # ticks_loc = ax1.get_yticks().tolist()
        # ax1.set_yticks(ax1.get_yticks().tolist())
        # ax1.set_yticklabels([x for x in ticks_loc], ha='right')
        #         label_format = '{:,.4f}'
        #         ax1.set_yticklabels([label_format.format(x) for x in ticks_loc], ha='right')

        if self.filebase:
            plt.savefig(self.filebase + title + ".png", format='png', bbox_inches='tight', pad_inches=0.01, dpi=300,
                        transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()
        return

    def energy(self, log=False, axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        plt.title(r"\noindent\textbf{" + "Energy" + "}", x=1, y=1.075)
        plt.xlabel(r"\textit{" + "Time" + "}")
        if axis:
            plt.axis(axis)

        for key, value in kwargs.items():
            if postprocess:
                value = postprocess(value)
            if isinstance(value, list):
                plt.plot(*value, linestyle='-', color="#E2365B", label=key)
            else:
                plt.plot(value, linestyle='-', color="#E2365B", label=key)

        handles, labels = ax1.get_legend_handles_labels()
        order = np.arange(len(handles))
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc=2, bbox_to_anchor=(-0.02, 1.2), frameon=False, ncol=4, columnspacing=1, fontsize=8)

        if self.filebase:
            plt.savefig(self.filebase + "energy.png", format='png', bbox_inches='tight', pad_inches=0.01, dpi=300,
                        transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()
        return


# def plot_correlation(f, g):
#     plt.style.use('../../ecostyle.mplstyle')
#     fig, ax1 = plt.subplots()
#     plt.xlabel(r"\textit{" + "r" + "}")
#     title = r"\noindent\textbf{" + "Autocorrelation function" + "}"
#     plt.title(title)
#
#     plt.plot(f, color="#595959", label="longitudinal")
#     plt.plot(g, color="#36E2BD", label="transversal")
#
#     handles, labels = ax1.get_legend_handles_labels()
#     order = np.arange(len(handles))
#     ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
#                loc=2, bbox_to_anchor=(-0.02, 1.2), frameon=False, ncol=1, columnspacing=1, fontsize=8)
#     plt.show()

# def plot_spectrum(x, w, wn=None, wm=None, *args):
#     energy = w.sum() if wn is None else wn.sum()
#     plt.style.use('../../ecostyle.mplstyle')
#     fig, ax1 = plt.subplots()
#     plt.xlabel(r"\textit{"+"Wavenumber"+"}")
#     title = r"\noindent\textbf{"+"Energy spectrum"+"}" + r"\\" + f"{energy:0.5f}"
#     #if i is None else f"{i+1:6.0f}"+r"\textbf{"+" Approximation"+"}"
#     plt.title(title)
#     plt.axis([3e-1,3e1,1e-8,1e1])
#     plt.plot(x[2:-3],(.1*x**(-5/3))[2:-3],linestyle='--',linewidth=0.75,color="#595959",label="")
# #     plt.plot(torch.arange(len(w))+1,w,linestyle='-',linewidth=0.75,color="#595959",label="Initial")
# #     if wn is not None:
# #         plt.plot(torch.arange(len(wn))+1,wn,linestyle='-',linewidth=1.5,color="#36E2BD",label="Instant")
# #     if wm is not None:
# #         plt.plot(torch.arange(len(wm))+1,wm,linestyle='-',linewidth=1,color="#E2365B",label="Average")
#     plt.plot(x,w,linestyle='-',linewidth=0.75,color="#595959",label="Initial")
#     if wn is not None:
#         plt.plot(x,wn,linestyle='-',linewidth=1.5,color="#36E2BD",label="Instant")
#     if wm is not None:
#         plt.plot(x,wm,linestyle='-',linewidth=1,color="#E2365B",label="Average")
# #     for wargs in args:
# #         plt.plot(torch.arange(len(wargs)),wargs,linestyle='-',linewidth=1,color="orange",label="old")
#     plt.xscale('log')
#     plt.yscale('log')
#     handles, labels = ax1.get_legend_handles_labels()
#     order = np.arange(len(handles))
#     ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#               loc=2, bbox_to_anchor=(-0.02, 1.2), frameon=False, ncol=1, columnspacing=1, fontsize=8)


# # --- Plot time_integral_max ---
# vel = torch.stack(velocity).cpu()
# # print(_s)
# # s = torch.stack(_s)
# #vel = torch.stack(vell)
# print(vel.shape)
# idnr = 0
# a_s = torch.stack(([torch.mean(vel[idnr]*vel[idnr+s]) for s in range(vel.shape[0]-idnr)]))/torch.mean(vel[idnr]**2)
#
# fig, ax1 = plt.subplots()
# fig.set_facecolor("white")
# plt.xlabel(r"\textit{"+"s [lu]"+"}")
# title = r"\noindent\textbf{"+"Two Point Correlation"+"}\n ar(s) \n"
# plt.title(title)
#
# ylabels= ([0,0.25,0.5,0.75,1])
# ax1.set_yticks(ylabels)
# # ax1.set_yticks([3000, 5000, 7000, 9000])
# ax1.set_yticklabels(ylabels, ha='right')
#
# plt.plot(torch.tensor(_s)[idnr:],a_s[:],label="Autokorrelationskoeffizient")
# handles, labels = ax1.get_legend_handles_labels()
# order = np.arange(len(handles))
# ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#               loc=2, bbox_to_anchor=(-0.02, 1.25), frameon=False, ncol=1, columnspacing=1, fontsize=8)
# file = filebase +"/autokorrelationskoeffizient-"+str(sys.argv[1][5:])+".png"
# plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
# plt.close()

# # ---plot correlation ---
# fig, ax1 = plt.subplots()
# fig.set_facecolor("white")
# plt.xlabel(r"\textit{"+"r"+"}")
# title = r"\noindent\textbf{"+"Two Point Correlation"+"}\n f(r), g(r) \n"
# plt.title(title)
#
# ylabels= ([0,0.25,0.5,0.75,1])
# ax1.set_yticks(ylabels)
# # ax1.set_yticks([3000, 5000, 7000, 9000])
# ax1.set_yticklabels(ylabels, ha='right')
#
# nr = 50
# plt.plot((np.stack(R11)[-nr:]).mean(0),label="longitudinal")
# plt.plot((np.stack(R22)[-nr:]).mean(0),label="transversal")
# handles, labels = ax1.get_legend_handles_labels()
# order = np.arange(len(handles))
# ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#               loc=2, bbox_to_anchor=(-0.02, 1.25), frameon=False, ncol=1, columnspacing=1, fontsize=8)
# file = filebase +"/correlation-"+str(sys.argv[1][5:])+".png"
# plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
# plt.close()