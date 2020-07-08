import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

class Solution:

    SPLIT_JV = 100

    def __init__(self, data_location, sol_number):

        """
        A class object which holds formatted data from IonMonger solution files. Designed to interact with an N by 2 cell of solution files produced by the batch_run.m
        script.

        Arguments:

            data_location: Location of batch results file
            sol_number: number of solution files in batch file

        Attributes:

            SPLIT_JV: number of temporal grid points in each stage of the simulation voltage program
            raw: Data in scipy.io.loadmat imported format
            label: Variable from batch run
            dat: Solution file
            dstrbns: Distribution data
            vectors: Space data
            params: Simulation parameters
            j: Current density data
            v: Voltage data
            widthP: Perovskite width
            stages: Number of stages in voltage program
            stage: list of integers used to slice data into stages
            nm2m: conversion parameter between nm^-3 and m^-3
            revXXX: data from reverse sweep
            fwdXXX: data from forward sweep
            keyval: list of integers showing location of RevVoc, RevMpp, Jsc, FwdMpp, and FwdVoc
            ndat: electron density data
            pdat: hole density data
            phi: electric potential data
            ionv: ion vacancy density data
            xxxE: data from ETL
            xxxH: data from HTL
            xxxP: data from perovskite layer
            DATAx: spatial data

        """

        # Format data
        self.raw = sio.loadmat(data_location)
        self.label = self.raw['results'][sol_number, 0][0][0]
        self.dat = self.raw['results'][sol_number, 1]
        self.dstrbns = self.dat['dstrbns'][0][0][0]
        self.vectors = self.dat['vectors'][0][0][0]
        self.params = self.dat['params'][0][0][0]
        self.j = self.dat['J'][0][0].flatten()
        self.v = self.dat['V'][0][0].flatten()

        # Simulation info
        self.widthP = self.params['b'][0][0][0]*1e9
        self.stages = int((len(self.params['applied_voltage'][0][0])-1)/3)
        self.stage = [i*self.SPLIT_JV for i in range(self.stages+1)]

        # Conversion parameter
        self.nm2m = 1e27

        # Assign RevVoc, RevMpp, Jsc, FwdMpp, FwdVoc
        self.revjdat = self.j[self.stage[-3]:self.stage[-2]]
        self.revvdat = self.v[self.stage[-3]:self.stage[-2]]

        self.revj0 = min(self.revjdat, key=lambda x:abs(x-0))
        self.RevVoc = np.where(self.revjdat == self.revj0)[0][0]

        self.fwdjdat = self.j[self.stage[-2]:self.stage[-1]]
        self.fwdvdat = self.v[self.stage[-2]:self.stage[-1]]

        self.fwdj0 = min(self.fwdjdat, key=lambda x:abs(x-0))
        self.FwdVoc = np.where(self.fwdjdat == self.fwdj0)[0][0]

        self.Jsc = self.stage[-2]

        self.revpwdat = np.multiply(self.revjdat, self.revvdat)
        self.RevMpp = np.where(self.revpwdat == np.amax(self.revpwdat))[0][0]

        self.fwdpwdat = np.multiply(self.fwdjdat, self.fwdvdat)
        self.FwdMpp = np.where(self.fwdpwdat == np.amax(self.fwdpwdat))[0][0]

        self.keyval = [self.RevVoc, self.RevMpp, self.Jsc, self.FwdMpp, self.FwdVoc]

        # Electron concentration
        self.ndatP = [self.dstrbns['n'][0][i,:]*self.nm2m for i in self.keyval]
        self.ndatE = [self.dstrbns['nE'][0][i,:]*self.nm2m for i in self.keyval]
        self.ndat = [np.append(i, k) for i, k in zip(self.ndatE, self.ndatP)]

        # Hole concentration
        self.pdatP = [self.dstrbns['p'][0][i,:]*self.nm2m for i in self.keyval]
        self.pdatH = [self.dstrbns['pH'][0][i,:]*self.nm2m for i in self.keyval]
        self.pdat = [np.append(i, k) for i, k in zip(self.pdatP, self.pdatH)]

        # Electric potential
        self.phiP = [self.dstrbns['phi'][0][i,:] for i in self.keyval]
        self.phiE = [self.dstrbns['phiE'][0][i,:] for i in self.keyval]
        self.phiH = [self.dstrbns['phiH'][0][i,:] for i in self.keyval]
        self.phiEP = [np.append(i, k) for i, k in zip(self.phiE, self.phiP)]
        self.phi = [np.append(i, k) for i, k in zip(self.phiEP, self.phiH)]

        # Ion vacancy density
        self.ionv = [self.dstrbns['P'][0][i,:]*self.nm2m for i in self.keyval]

        # Electron x data
        self.nxP = self.vectors['x'][0]*self.widthP
        self.nxE = self.vectors['xE'][0]*self.widthP
        self.nx = np.append(self.nxE, self.nxP)

        # Hole x data
        self.pxP = self.vectors['x'][0]*self.widthP
        self.pxH = self.vectors['xH'][0]*self.widthP
        self.px = np.append(self.pxP, self.pxH)

        # Electric potential x data
        self.phix = np.append(np.append(self.nxE, self.nxP), self.pxH)

        # Ion vacancy x data
        self.ionvx = self.nxP

def plot_electronsholes(solution, save=False):

    """
    Function that plots the electron and hole distributions when given a Solution object.

    Arguments:

        solution: Solution class object
        save: set to True to save the figure

    """

    # Create figure and axes

    fig, ax = plt.subplots()

    # Graph dimensions for visual elements
    gh = np.append(solution.ndat, solution.pdat)
    bot = min(gh)
    top = max(gh)
    boxy = np.asarray([bot,bot,top,top], dtype=object)

    leftE = solution.nx[0]
    rightE = solution.pxP[0]

    leftH = solution.pxP[-1]
    rightH = solution.px[-1]

    # Plot shaded regions
    ax.fill(np.asarray([leftE,rightE,rightE,leftE], dtype=object), boxy, 'C0', alpha = 0.2)
    ax.fill(np.asarray([leftH,rightH,rightH,leftH], dtype=object), boxy, 'C3', alpha = 0.2)

    # Add text
    ax.text(np.median(solution.nxE), np.median(gh), 'ETL', c='C0')
    ax.text(np.median(solution.pxH), np.median(gh), 'HTL', c='C3')

    # Legends and lines
    ax.plot(0,0, label = 'Electrons', c = 'C0')
    ax.plot(0,0, label = 'Holes', c = 'C3')

    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']
    lnsty = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
    for i, j in zip(lgndkeyval, lnsty):
        ax.plot(0,0, label = i, ls = j, c = 'k')

    ax.legend()

    # Plot data
    for i, j in zip(solution.ndat, lnsty):
        ax.plot(solution.nx, i, c='C0', ls = j)

    for i, j in zip(solution.pdat, lnsty):
        ax.plot(solution.px, i, c='C3', ls = j)

    # Make y axis log scale
    ax.set_yscale('log')
    ax.set_title(f'Charge carrier distributions for {solution.label}')
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Carrier concentration (m$^{-3}$)')

    # Save file:
    if save == True:
        fig.savefig(f'np_dstrbn_{solution.label}.png', dpi = 400)




def plot_electricpotential(solution, save=False):

    """
    Function that plots the electric potential distribution when given a Solution object.

    Arguments:

        solution: Solution class object
        save: set to True to save the figure

    """

    # Create figure and axes
    fig, ax = plt.subplots()

    # Graph dimensions for visual elements
    gh = np.asarray(solution.phi, dtype=object).flatten()
    bot = min(gh)
    top = max(gh)
    boxy = np.asarray([bot,bot,top,top], dtype=object)

    leftE = solution.nx[0]
    rightE = solution.pxP[0]

    leftH = solution.pxP[-1]
    rightH = solution.px[-1]

    # Plot shaded regions
    ax.fill(np.asarray([leftE,rightE,rightE,leftE], dtype=object), boxy, 'C0', alpha = 0.2)
    ax.fill(np.asarray([leftH,rightH,rightH,leftH], dtype=object), boxy, 'C3', alpha = 0.2)

    # Add text
    ax.text(np.median(solution.nxE), np.median(gh), 'ETL', c='C0')
    ax.text(np.median(solution.pxH), np.median(gh), 'HTL', c='C3')


    # Legends and lines
    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']

    # Plot data
    for i, j in zip(solution.phi, lgndkeyval):
        ax.plot(solution.phix, i, label = j)

    # Plot legend and axes labels
    ax.legend()
    ax.set_title(f'Electric potential distribution for {solution.label}')
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Electric potential (V)')

    # Save file:
    if save == True:
        fig.savefig(f'pot_dstrbn_{solution.label}.png', dpi = 400)




def plot_anionvacancies(solution, save=False):

    """
    Function that plots the anion vacancy distribution when given a Solution object.

    Arguments:

        solution: Solution class object
        save: set to True to save the figure

    """

    # Create figure and axes
    fig, ax = plt.subplots()

    # Legends and lines
    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']

    # Plot data
    for i, j in zip(solution.ionv, lgndkeyval):
        ax.plot(solution.ionvx, i, label = j)

    # Plot legend and axes labels
    ax.legend()
    ax.set_yscale('log')
    ax.set_title(f'Anion vacancy distribution for {solution.label}')
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Anion vacancy density (m$^{-3}$)')

    # Save file:
    if save == True:
        fig.savefig(f'anion_vac_dstrbn_{solution.label}.png', dpi = 400)




def plot_zoomed_anionvacancies(solution, transportlayer, save=False, zoom=125):

    """
    Function that plots the anion vacancy distribution at a transport layer/perovskite interface when given a Solution object.

    Arguments:

        solution: Solution class object
        transportlayer: set to 'E' for electron transport layer or 'H' for hole transport layer
        save: set to True to save the figure
        zoom: degree of zoom, uses grid of solution file (default=125)

    """

    # Create figure and axes
    fig, ax = plt.subplots()

    # Legends and lines
    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']

    # Plot data
    for i, j in zip(solution.ionv, lgndkeyval):
        if transportlayer == 'E':
            ax.plot(solution.ionvx[0:zoom], i[0:zoom], label = j)
        elif transportlayer == 'H':
            ax.plot(solution.ionvx[-zoom:-1], i[-zoom:-1], label = j)

    # Plot legend and axes labels
    ax.legend()
    ax.set_yscale('log')
    if transportlayer == 'E':
        ax.set_title(f'Anion vacancy distribution for {solution.label} at ETL')
    elif transportlayer == 'H':
        ax.set_title(f'Anion vacancy distribution for {solution.label} at HTL')
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Anion vacancy density (m$^{-3}$)')

    # Save file:
    if save == True:
        if transportlayer == 'E':
            fig.savefig(f'ETL_anion_vac_dstrbn_{solution.label}.png', dpi = 400)
        elif transportlayer == 'H':
            fig.savefig(f'HTL_anion_vac_dstrbn_{solution.label}.png', dpi = 400)
