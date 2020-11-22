# Module to load and plot jv data

import os
import numpy as np
import matplotlib.pyplot as plt

def load_file_paths(directory):

    '''
    Returns a recursive list of file paths when given a target directory.
    '''

    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_paths.append(os.path.join(root, file))
    return file_paths

def stats_dictionary(file_paths, size):

    '''
    Returns a dictionary of pixel statistics dictionaries when given a list of
    file paths and the size of the pixels.

    Dictionary structured as:
        {pixel:
        '{'jsc': jsc, 'mean_pce': mean_pce, 'hysteresis': hysteresis,
        'mean_voc': mean_voc, 'mean_ff': mean_ff,
        'reverse': {'voc': reverse_voc, 'ff': reverse_ff, 'pce': reverse_pce},
        'forward': {'voc': forward_voc, 'ff': forward_ff, 'pce': forward_pce}}'
        }
    '''

    stats_dictionary = {}
    for i in file_paths:
        jv = JvData(i)
        stats = jv.calculate_stats(size)
        stats_dictionary[f'{jv.label}'] = stats
    return stats_dictionary

class JvData:

    '''
    Returns an object with a label attribute and data array attribute.
    '''

    def __init__(self, path, delimiter=',', string_slice=(-4, -19)):
        self.data = np.genfromtxt(path, delimiter=delimiter)
        self.label = path[string_slice[1]:string_slice[0]]

    def plot_jv(self, save=False, setax=False):

        '''
        Plots a graph of current density vs voltage.
        '''

        half = int(len(self.data[:, 1]) / 2)

        fig, ax = plt.subplots()
        ax.plot(self.data[:half, 0], self.data[:half, 1],
                label='Reverse', color='C0', linestyle='-')
        ax.plot(self.data[half:, 0], self.data[half:, 1],
                label='Forward', color='C1', linestyle='--')

        ax.legend()
        ax.set_ylabel('Current density (mA cm$^{-2}$)')
        ax.set_xlabel('Voltage (V)')

        # Axis control
        if setax is not False:
            ax.set_xlim(setax[0][0], setax[0][1])
            ax.set_ylim(setax[1][0], setax[1][1])

        # Save file:
        if save is True:
            fig.savefig(f'jv_{self.label}.png', dpi=400)

    def calculate_stats(self, size):

        '''
        Returns a dictionary of pixel statistics.

        Dictionary structured as:
            '{'jsc': jsc, 'mean_pce': mean_pce, 'hysteresis': hysteresis,
            'mean_voc': mean_voc, 'mean_ff': mean_ff,
            'reverse': {'voc': reverse_voc, 'ff': reverse_ff, 'pce': reverse_pce},
            'forward': {'voc': forward_voc, 'ff': forward_ff, 'pce': forward_pce}}
        '''

        half = int(len(self.data[:, 1]) / 2)

        jsc = self.data[half, 1]

        reverse_j0 = min(self.data[:half, 1], key=lambda x:abs(x-0))
        reverse_voc_index = np.where(self.data[:half, 1] == reverse_j0)[0][0]
        reverse_voc = self.data[reverse_voc_index, 0]

        forward_j0 = min(self.data[half:, 1], key=lambda x:abs(x-0))
        forward_voc_index = np.where(self.data[half:, 1] == forward_j0)[0][0] + half
        forward_voc = self.data[forward_voc_index, 0]

        mean_voc = (reverse_voc + forward_voc) / 2

        power_in = 0.1 * size
        power = self.data[:, 0] * self.data[:, 1]

        reverse_mpp = max(power[:half])
        reverse_mpp_index = np.where(power[:half] == reverse_mpp)[0][0]
        reverse_vmp = self.data[reverse_mpp_index, 0]
        reverse_jmp = self.data[reverse_mpp_index, 1]
        reverse_ff = reverse_vmp * reverse_jmp / reverse_voc / jsc

        forward_mpp = max(power[half:])
        forward_mpp_index = np.where(power[half:] == forward_mpp)[0][0] + half
        forward_vmp = self.data[forward_mpp_index, 0]
        forward_jmp = self.data[forward_mpp_index, 1]
        forward_ff = forward_vmp * forward_jmp / forward_voc / jsc

        mean_ff = (reverse_ff + forward_ff) / 2

        reverse_pce = reverse_vmp * reverse_jmp * size / 1e3 / power_in
        forward_pce = forward_vmp * forward_jmp * size / 1e3 / power_in
        mean_pce = (reverse_pce + forward_pce) / 2

        reverse_area = np.trapz(self.data[reverse_voc_index:half, 1],
                                x=-self.data[reverse_voc_index:half, 0])
        forward_area = np.trapz(self.data[half:forward_voc_index, 1],
                                x=self.data[half:forward_voc_index, 0])
        hysteresis = (reverse_area - forward_area) / reverse_area

        return {'jsc': jsc, 'mean_pce': mean_pce, 'hysteresis': hysteresis,
                'mean_voc': mean_voc, 'mean_ff': mean_ff,
                'reverse': {'voc': reverse_voc, 'ff': reverse_ff, 'pce': reverse_pce},
                'forward': {'voc': forward_voc, 'ff': forward_ff, 'pce': forward_pce}}
