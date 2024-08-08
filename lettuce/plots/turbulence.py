import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker

__all__ = [
    "Plot"
]
class Plot:
    def __init__(self, filebase=None, show=True, style=None):
        self.show = show
        self.filebase = filebase
        if style:
            plt.style.use(style)


    def __call__(self, f):
        raise NotImplementedError

    def basic(self, title='chart', xlabel='x', log=False, axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        # plt.title(r"\noindent\textbf{" + title + "}", x=1, y=1.075)
        # plt.xlabel(r"\textit{" + xlabel + "}")
        plt.title(title, x=1, y=1.075)
        plt.xlabel(r"\textit{" + f"{xlabel}" + "}")
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

    def correlation(self, f=None, g=None, a=None):
        # plt.style.use('../../ecostyle.mplstyle')
        fig, ax1 = plt.subplots()
        plt.title(r"\noindent\textbf{" + "Autocorrelation Function" + "}", loc='left', )
        ylabels = ([0, 0.25, 0.5, 0.75, 1])
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabels, ha='right')
        plt.ylim(-0.2, 1.2)
        if f is not None:
            plt.xlabel(r"\textit{" + "r" + "}")
            plt.plot(f, color="#595959", label=r"Longitudinal \textendash{} f(r)")
            plt.plot(g, color="#36E2BD", label=r"Transversal \textendash{} g(r)")
            dataname = "autocorrelation"
        if a:
            plt.xlabel(r"\textit{" + "s" + "}")
            plt.plot(a[0], a[1], color="#595959", label=r"Autokorrelationscoefficient \textendash{} ar(s)")
            dataname = "timecorrelation"

        handles, labels = ax1.get_legend_handles_labels()
        order = np.arange(len(handles))
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc=2, bbox_to_anchor=(-0.02, 1), frameon=False, ncol=2, columnspacing=2, fontsize=8)
        self._out(dataname)
        return

    def spectrum(self, title='Energy Spectrum', k53=False, k53_factor=1, axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        # plt.title(r"\noindent\textbf{" + title + "}")
        plt.title(title)
        # plt.xlabel(r"\textit{" + "Wavenumber" + "}")
        plt.xlabel("Wavenumber")
        plt.xscale('log')
        plt.yscale('log')
        ylabels = ([1e-4, 1e-2, 1e-0])
        plt.ylim([1e-5,1e1])
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabels, ha='right')
        if axis:
            plt.axis(axis)

        if k53 is not False:
            plt.plot(k53, (k53_factor * k53 ** (-5 / 3)), linestyle='--', linewidth=0.75, color="#595959", label="k53")

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
                   loc=2, bbox_to_anchor=(0, 1.14), frameon=False, ncol=2, columnspacing=1, fontsize=8)

        self._out(title)

    def dissipation(self, id=0, log=False, y_axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        plt.title(f"Dissipation_{id:05d}", x=1, y=1.075)
        plt.xlabel("Time")
        ylabels = ([0, 0.005, 0.01, 0.015, 0.02])
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabels, ha='right')
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        if y_axis:
            plt.ylim(y_axis)

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
                   loc=2, bbox_to_anchor=(-0.02, 1.2), frameon=False, ncol=4, columnspacing=1, fontsize=8)

        if self.filebase:
            plt.savefig(self.filebase + f"dissipation_{id:05d}.png", format='png', bbox_inches='tight', pad_inches=0.01,
                        dpi=300,
                        transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()
        return

    def training_dissipation(self, id=0, log=False, y_axis=None, postprocess=None, *args, **kwargs):
        fig, ax1 = plt.subplots()
        plt.title(f"Dissipation_{id:05d}", x=1, y=1.075)
        plt.xlabel("Time")
        ylabels = ([0, 0.005, 0.01, 0.015, 0.02])
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabels, ha='right')
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        if y_axis:
            plt.ylim(y_axis)

        for key, value in kwargs.items():
            if postprocess:
                value = postprocess(value)
            if 'Neural' in key:
                if isinstance(value, list):
                    plt.plot(*value, linestyle='-', lw=1, label=key, color='#F6423C')
                else:
                    plt.plot(value, linestyle='-', lw=1, label=key, color='#F6423C')
            if 'Reference' in key:
                if isinstance(value, list):
                    plt.plot(*value, linestyle='-', lw=1.5, label=key, color='#595959')
                else:
                    plt.plot(value, linestyle='-', lw=1.5, label=key, color='#595959')

        handles, labels = ax1.get_legend_handles_labels()
        labels = ['Reference', 'NCO']
        order = np.arange(len(handles))
        # ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
        ax1.legend([handles[idx] for idx in [0, 1]], [labels[idx] for idx in [0, 1]],
                   loc=2, bbox_to_anchor=(-0.02, 1.2), frameon=False, ncol=4, columnspacing=1, fontsize=8)

        if self.filebase:
            plt.savefig(self.filebase + f"dissipation_{id:05d}.png", format='png', bbox_inches='tight', pad_inches=0.01, dpi=300,
                        transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()
        return

    def _out(self, dataname="spectrum"):
        if self.filebase:
            plt.savefig(self.filebase + dataname +".png", format='png', bbox_inches='tight', pad_inches=0.01, dpi=300,
                        transparent=False)
        if self.show:
            plt.show()
        else:
            plt.close()




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
