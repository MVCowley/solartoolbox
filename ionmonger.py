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

            data_location: location of batch results file
            sol_number: index of solution file in batch file to access

        Attributes:

            SPLIT_JV: number of temporal grid points in each stage of the simulation voltage program
            raw: data in scipy.io.loadmat imported format
            label: variable from batch run
            dat: solution file
            dstrbns: distribution data
            vectors: space data
            params: simulation parameters
            j: current density data
            v: voltage data
            widthP: perovskite width
            stages: number of stages in voltage program
            stage: list of integers used to slice data into stages
            paramsdic: dictionary of simulation parameter labels and values
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
            xxxy: y data for np.trapz calculation
            xxxarea: area under xxx curve given by np.trapz
            degreehyst: degree of hysteresis in simulation

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
        self.paramsdic = {l:v.flatten() for l, v in zip(self.params[0].dtype.names, self.params[0])}

        # Conversion parameter
        self.nm2m = 1e27

        # Assign RevVoc, RevMpp, Jsc, FwdMpp, FwdVoc
        self.revjdat = self.j[self.stage[-3]:self.stage[-2]]
        self.revvdat = self.v[self.stage[-3]:self.stage[-2]]

        self.revj0 = min(self.revjdat, key=lambda x:abs(x-0))
        self.RevVoc = np.where(self.revjdat == self.revj0)[0][0] + self.stage[-3]

        self.fwdjdat = self.j[self.stage[-2]:self.stage[-1]]
        self.fwdvdat = self.v[self.stage[-2]:self.stage[-1]]

        self.fwdj0 = min(self.fwdjdat, key=lambda x:abs(x-0))
        self.FwdVoc = np.where(self.fwdjdat == self.fwdj0)[0][0] + self.stage[-2]

        self.Jsc = np.where(min(self.v) == self.v)[0][0]

        self.revpwdat = np.multiply(self.revjdat, self.revvdat)
        self.RevMpp = np.where(self.revpwdat == np.amax(self.revpwdat))[0][0] + self.stage[-3]

        self.fwdpwdat = np.multiply(self.fwdjdat, self.fwdvdat)
        self.FwdMpp = np.where(self.fwdpwdat == np.amax(self.fwdpwdat))[0][0] + self.stage[-2]

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

        # Degree of hysteresis calculation
        self.revy = self.j[self.RevVoc:self.Jsc+1]
        self.revarea = np.trapz(self.revy, dx = np.diff(-self.revvdat)[0])

        self.fwdy = self.j[self.Jsc:self.FwdVoc+1]
        self.fwdarea = np.trapz(self.fwdy, dx = np.diff(self.fwdvdat)[0])

        self.degreehyst = ((self.revarea - self.fwdarea) / self.revarea) * 100




def plot_electronsholes(solution, save=False, setax=False):

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

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'np_dstrbn_{solution.label}.png', dpi = 400)




def plot_electricpotential(solution, save=False, setax=False):

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

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'pot_dstrbn_{solution.label}.png', dpi = 400)




def plot_anionvacancies(solution, save=False, setax=False):

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

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'anion_vac_dstrbn_{solution.label}.png', dpi = 400)




def plot_zoomed_anionvacancies(solution, transportlayer, save=False, zoom=125, setax=False):

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

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        if transportlayer == 'E':
            fig.savefig(f'ETL_anion_vac_dstrbn_{solution.label}.png', dpi = 400)
        elif transportlayer == 'H':
            fig.savefig(f'HTL_anion_vac_dstrbn_{solution.label}.png', dpi = 400)



def latex_table(solution):

    '''
    Function to print simulation parameters in LaTeX table format. Imports pandas.
    '''

    import pandas as pd
    parameters_df = pd.DataFrame(data=[solution.paramsdic]).T
    print(parameters_df.to_latex())

def plot_degree_of_hysteresis(solution_batch, precondition, title, save=False, setax=False):
    scan_rate = [precondition/i.label for i in solution_batch]
    degreehyst = [i.degreehyst for i in solution_batch]

    fig, ax = plt.subplots()

    ax.plot(scan_rate, degreehyst, marker='o', markersize=3)
    ax.set_xscale('log')
    ax.set_xlabel('Scan rate mV (s$^{-1}$)')
    ax.set_ylabel('Degree of hysteresis (%)')
    ax.set_title(f'Degree of hysteresis vs scan rate for {title}')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'hysteresis_scan_rate_{title}.png', dpi = 400)

def plot_electric_force_scan_rate(solution_batch, label_modifier, point_of_interest, title, save=False, setax=False):
    electric_force = [-(np.diff(i.phiP)/1e6)/(i.vectors['dx'][0].flatten()*i.widthP/1e9) for i in solution_batch]
    scan_rate = [label_modifier/i.label for i in solution_batch]

    revvoc = []
    revmpp = []
    jsc = []
    fwdmpp = []
    fwdvoc = []
    middle_force = [revvoc, revmpp, jsc, fwdmpp, fwdvoc]

    for solution_array in electric_force:
        revvoc.append(solution_array[0][point_of_interest])
        revmpp.append(solution_array[1][point_of_interest])
        jsc.append(solution_array[2][point_of_interest])
        fwdmpp.append(solution_array[3][point_of_interest])
        fwdvoc.append(solution_array[4][point_of_interest])

    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']

    fig, ax = plt.subplots()

    for i, j in zip(middle_force, lgndkeyval):
        ax.plot(scan_rate, i, marker='o', markersize=3, label=j)

    ax.legend()
    ax.set_xscale('log')
    ax.set_ylabel('Electric field (MV m$^{-1}$)')
    ax.set_xlabel('Scan rate (mV s$^{-1}$)')
    ax.set_title(f'Electric force vs scan rate for {title} at {point_of_interest}')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'electric_force_scan_rate_{title}_{point_of_interest}.png', dpi = 400)

def intrinsic_carriers(solution):

    nc = solution.paramsdic['gc']
    nv = solution.paramsdic['gv']
    ec = solution.paramsdic['Ec']
    ev = solution.paramsdic['Ev']
    kb = solution.paramsdic['kB']
    T = solution.paramsdic['T']

    i = np.sqrt(nc*nv*np.exp(-(ec-ev)/(kb*T)))
    return i

def srh_recombination_rate(solution):

    steps = len(solution.dstrbns['n'][0])

    electrons = [solution.dstrbns['n'][0][i,:]*solution.nm2m for i in range(1, steps)]
    holes = [solution.dstrbns['p'][0][i,:]*solution.nm2m for i in range(1, steps)]

    i = intrinsic_carriers(solution)
    t_n = solution.paramsdic['tn']
    t_p = solution.paramsdic['tp']

    srh = [(n*p-i**2)/(t_n*p + t_p*n + (t_n + t_p)*i) for n, p in zip(electrons, holes)]
    return srh

def plot_srh_scan_rate(batch_solution, label_modifier, point_of_interest, title, save=False, setax=False):
    scan_rate = [label_modifier/i.label for i in batch_solution]
    srh = [srh_recombination_rate(i) for i in batch_solution]

    revvoc = []
    revmpp = []
    jsc = []
    fwdmpp = []
    fwdvoc = []
    middle_srh= [revvoc, revmpp, jsc, fwdmpp, fwdvoc]

    for index, solution_array in enumerate(srh):
        revvoc.append(solution_array[batch_solution[index].keyval[0]][point_of_interest])
        revmpp.append(solution_array[batch_solution[index].keyval[1]][point_of_interest])
        jsc.append(solution_array[batch_solution[index].keyval[2]][point_of_interest])
        fwdmpp.append(solution_array[batch_solution[index].keyval[3]][point_of_interest])
        fwdvoc.append(solution_array[batch_solution[index].keyval[4]][point_of_interest])

    lgndkeyval = ['RevVoc', 'RevMpp', 'Jsc', 'FwdMpp', 'FwdVoc']
    colours = ['C0', 'C1', 'C2', 'C1', 'C0']
    linestyle = ['dashed', 'dashed', 'solid', 'dotted', 'dotted']

    fig, ax = plt.subplots()

    for i, j, k, l in zip(middle_srh, lgndkeyval, colours, linestyle):
        ax.plot(scan_rate, i, marker='o', markersize=3, label=j, c=k, ls=l)

    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Scan rate (mV s$^{-1}$)')
    ax.set_yscale('log')
    ax.set_ylabel('SRH recombination (m$^{-3}$ s$^{-2}$)')
    ax.set_title(f'SRH recombination vs scan rate for {title} at {point_of_interest}')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'srh_recombination_scan_rate_{title}_{point_of_interest}.png', dpi = 400)

def plot_jv_curve(solution, precondition, save=False, setax=False):

    fig, ax = plt.subplots()

    ax.plot(solution.v[solution.RevVoc:solution.Jsc+1], solution.j[solution.RevVoc:solution.Jsc+1], label='Rev', color='C0', linestyle='-')
    ax.plot(solution.v[solution.Jsc:solution.FwdVoc+1], solution.j[solution.Jsc:solution.FwdVoc+1], label='Fwd', color='C1', linestyle='--')

    ax.legend()
    ax.set_ylabel('Current density (mA cm$^{-2}$)')
    ax.set_xlabel('Voltage (V)')
    ax.set_title(f'jV curve for {precondition/solution.label}' + 'mV s$^{-1}$')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'jv_{precondition/solution.label}.png', dpi = 400)

def plot_currents(solution_batch, label_modifier, title, save=False, setax=False, sims=50):

    j_rev = [solution_batch[i].j[solution_batch[i].RevMpp] for i in range(sims)]
    jl_rev = [-solution_batch[i].dat['Jl'].flatten()[0][solution_batch[i].RevMpp] for i in range(sims)]
    jr_rev = [-solution_batch[i].dat['Jr'].flatten()[0][solution_batch[i].RevMpp] for i in range(sims)]

    j_fwd = [solution_batch[i].j[solution_batch[i].FwdMpp] for i in range(sims)]
    jl_fwd = [-solution_batch[i].dat['Jl'].flatten()[0][solution_batch[i].FwdMpp] for i in range(sims)]
    jr_fwd = [-solution_batch[i].dat['Jr'].flatten()[0][solution_batch[i].FwdMpp] for i in range(sims)]

    scan_rate = [label_modifier/solution_batch[i].label for i in range(sims)]

    fig, ax = plt.subplots()

    ax.plot(scan_rate, j_rev, color='g')
    ax.plot(scan_rate, jl_rev, color='b')
    ax.plot(scan_rate, jr_rev, color='r')

    ax.plot(scan_rate, j_fwd, color='g', linestyle='dashed')
    ax.plot(scan_rate, jl_fwd, color='b', linestyle='dashed')
    ax.plot(scan_rate, jr_fwd, color='r', linestyle='dashed')

    current = ['Photocurrent', 'ETL recombination', 'HTL recombination']
    colour = ['g', 'b', 'r']
    for i, j in zip(current, colour):
        ax.plot(0, 0, label=i, c=j)

    direction = ['Reverse scan', 'Forward scan']
    linestyle = ['solid', 'dashed']
    for i, j in zip(direction, linestyle):
        ax.plot(0, 0, label=i, linestyle=j, color='k')

    ax.legend()
    ax.set_title(title)

    ax.set_xscale('log')
    ax.set_xlabel('Scan rate (mV/s)')

    ax.set_yscale('log')
    ax.set_ylabel('Current density (mA/cm$^2$)')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'currents_scan_rate_{title}.png', dpi = 400)
