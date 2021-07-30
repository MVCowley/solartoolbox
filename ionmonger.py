import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'

class BatchData:

    def __init__(self, data_location):

        """

        A class object which loads in a IonMonger batch file.

        """

        self.raw = sio.loadmat(file_name=data_location)['results']

class Solution:

    SPLIT_JV = 100

    def __init__(self, data, label=None):

        """
        A class object which holds formatted data from IonMonger solution files. Designed to be called from
        either classmethod from_batch() or from_single().

        Arguments:

            data: data from scipy.io.loadmat()
            label: index provided by from_batch() for scan rate plots

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

        # Load data from factory methods
        self.label = label
        self.dat = data
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
        self.n0 = self.paramsdic['N0'][0]
        self.dE = self.paramsdic['dE'][0]
        self.dH = self.paramsdic['dH'][0]
        self.vt = self.paramsdic['VT'][0]

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
        self.ndatP = [self.dstrbns['n'][0][i,:]*self.dE for i in self.keyval]
        self.ndatE = [self.dstrbns['nE'][0][i,:]*self.dE for i in self.keyval]
        self.ndat = [np.append(i, k) for i, k in zip(self.ndatE, self.ndatP)]

        # Hole concentration
        self.pdatP = [self.dstrbns['p'][0][i,:]*self.dH for i in self.keyval]
        self.pdatH = [self.dstrbns['pH'][0][i,:]*self.dH for i in self.keyval]
        self.pdat = [np.append(i, k) for i, k in zip(self.pdatP, self.pdatH)]

        # Electric potential
        self.phiP = [self.dstrbns['phi'][0][i,:]*self.vt for i in self.keyval]
        self.phiE = [self.dstrbns['phiE'][0][i,:]*self.vt for i in self.keyval]
        self.phiH = [self.dstrbns['phiH'][0][i,:]*self.vt for i in self.keyval]
        self.phiEP = [np.append(i, k) for i, k in zip(self.phiE, self.phiP)]
        self.phi = [np.append(i, k) for i, k in zip(self.phiEP, self.phiH)]

        # Ion vacancy density
        self.ionv = [self.dstrbns['P'][0][i,:]*self.n0 for i in self.keyval]

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
        self.revarea = np.trapz(self.revy, x=-self.v[self.RevVoc:self.Jsc+1])

        self.fwdy = self.j[self.Jsc:self.FwdVoc+1]
        self.fwdarea = np.trapz(self.fwdy, x=self.v[self.Jsc:self.FwdVoc+1])

        self.degreehyst = ((self.revarea - self.fwdarea) / self.revarea) * 100

    @classmethod
    def from_batch(cls, batch_data, sol_number):
        """
        If creating Solution objects from the batch script in the readme of solartoolbox, use this
        classmethod.

        batch_data: class object supplied by BatchData()
        sol_number: index of solution file in batch file to access

        Can create a list of solution objects via:
            batch_load = BatchData(batch)
            batch_sol = [Solution.from_batch(batch_load, i) for i in range(len(batch_load.raw))]
        """
        label = batch_data.raw[sol_number, 0][0][0]
        dat = batch_data.raw[sol_number, 1]
        return cls(data=dat, label=label)

    @classmethod
    def from_single(cls, file_location, key='sol', within_struct=False):
        """
        If creating a Solution object from a single sol output of IonMonger, use this classmethod.
        """
        if within_struct is True:
            data=sio.loadmat(file_name=file_location)[key][0][0]
        else:
            data=sio.loadmat(file_name=file_location)[key]
        return cls(data)

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




def plot_electricpotential(solution, title, save=False, setax=False):

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
    ax.set_title(f'Electric potential distribution for {title}')
    ax.set_xlabel('Thickness (nm)')
    ax.set_ylabel('Electric potential (V)')

    # Axis control
    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'pot_dstrbn_{title}.png', dpi = 400)




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

    ax.plot(scan_rate, degreehyst, marker='o', markersize=3, c='m')
    ax.set_xscale('log')
    ax.set_xlabel('Scan rate (mV s$^{-1}$)')
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
    colours = ['C0', 'C1', 'C2', 'C1', 'C0']
    linestyle = ['dashed', 'dashed', 'solid', 'dotted', 'dotted']

    fig, ax = plt.subplots()

    for i, j, k, l in zip(middle_force, lgndkeyval, colours, linestyle):
        ax.plot(scan_rate, i, label=j, c=k, ls=l)

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

def srh_recombination_rate(solution):

    n = solution.dstrbns['n'][0]
    p = solution.dstrbns['p'][0]
    ni2 = solution.paramsdic['ni2']
    gamma = solution.paramsdic['gamma']
    tor = solution.paramsdic['tor']
    tor3 = solution.paramsdic['tor3']

    rescale = solution.paramsdic['G0']

    srh = gamma*(p-ni2/n)/(1+tor*p/n+tor3/n)*(n>=tor*p)*(n>=tor3) \
        + gamma*(n-ni2/p)/(n/p+tor+tor3/p)*(tor*p>n)*(tor*p>tor3) \
        + gamma*(p*n-ni2)/(n+tor*p+tor3)*(tor3>n)*(tor3>=tor*p)

    return srh * rescale

def bimolecular_recombination(solution):

    n = solution.dstrbns['n'][0]
    p = solution.dstrbns['p'][0]
    ni2 = solution.paramsdic['ni2']
    brate = solution.paramsdic['brate']
    rescale = solution.paramsdic['G0']

    bimolecular = brate * (n*p-ni2)

    return bimolecular * rescale

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
        ax.plot(scan_rate, i, label=j, c=k, ls=l)

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

def plot_anion_vac_change(solution_batch, label_modifier, title, zoom=125, setax=False, save=False):
    scan_rate = [label_modifier/i.label for i in solution_batch]

    etl_data = []
    htl_data = []

    for i in solution_batch:
        etl_anion_vacancies = np.asarray(i.ionv, dtype=object)[:,0:zoom]
        etl_change = etl_anion_vacancies.max() - etl_anion_vacancies.min()
        etl_data.append(etl_change)

        htl_anion_vacancies = np.asarray(i.ionv, dtype=object)[:,-zoom:-1]
        htl_change = htl_anion_vacancies.max() - htl_anion_vacancies.min()
        htl_data.append(htl_change)

    fig, ax = plt.subplots()

    ax.plot(scan_rate, etl_data, c='b', label='Vacancy change at ETL')
    ax.plot(scan_rate, htl_data, c='r', label='Vacancy change at HTL')

    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Scan rate (mV s$^{-1}$)')
    ax.set_yscale('log')
    ax.set_ylabel('Change in anion vacancy density (m$^{-3}$)')
    ax.set_title(f'Vacancy delta during scan for {title}')

    if setax is not False:
        ax.set_xlim(setax[0][0], setax[0][1])
        ax.set_ylim(setax[1][0], setax[1][1])

    # Save file:
    if save == True:
        fig.savefig(f'anion_change_{title}.png', dpi = 400)

def for_x_nondim_species_current(solution, species, x):

    density = solution.dstrbns[species][0]
    phi = solution.dstrbns['phi'][0]
    dx = solution.vectors['dx'][0].flatten()
    mid = x

    if species == 'n':
        species_param = solution.params['Kn'][0][0][0]
        species_current = species_param / dx[mid] * (
            density[:,mid+1] - density[:,mid]
            - ( density[:,mid+1] + density[:,mid] )
            * ( phi[:,mid+1] - phi[:,mid] ) / 2 )

    elif species == 'p':
        species_param = solution.params['Kp'][0][0][0]
        species_current = - species_param / dx[mid] * (
            density[:,mid+1] - density[:,mid]
            + ( density[:,mid+1] + density[:,mid] )
            * ( phi[:,mid+1] - phi[:,mid] ) / 2 )

    elif species == 'P':
        species_param = solution.params['dpf'][0][0][0]
        species_current = - species_param / dx[mid] * (
            density[:,mid+1] - density[:,mid]
            + ( density[:,mid+1] + density[:,mid] )
            * ( phi[:,mid+1] - phi[:,mid] ) / 2 )
    else:
        print('Please enter either n, p, or P for species.')

    return species_current

def for_x_nondim_current(solution, x):

    jn = for_x_nondim_species_current(solution, 'n', x)
    jp = for_x_nondim_species_current(solution, 'p', x)
    jf = for_x_nondim_species_current(solution, 'P', x)

    time = solution.dat['time'][0][0][0]
    phi = solution.dstrbns['phi'][0]
    dx = solution.vectors['dx'][0].flatten()
    dt = np.diff(time)
    mid = x

    dis_param = solution.params['dpt'][0][0][0]
    jd = np.empty(shape=(301))
    jd[0] = None
    for i in range(1, len(time+1)):
        jd[i] = - dis_param / dx[mid] * (
            phi[i,mid+1] - phi[i,mid] - phi[i-1,mid+1] + phi[i-1,mid] )

    pbi = solution.params['pbi'][0][0][0]
    arp = solution.params['ARp'][0][0][0]
    jr = np.empty(shape=(301))
    for i in range(len(time)-1):
        jr[i] = ( pbi - ( solution.dstrbns['phiE'][0][i,0]
                         - solution.dstrbns['phiH'][0][i,-1] ) ) / arp

    current = jn + jp - jf - jd - jr

    return current

def for_x_seperate_nondim_current_npP(solution, x):

    jn = for_x_nondim_species_current(solution, 'n', x)
    jp = for_x_nondim_species_current(solution, 'p', x)
    jf = for_x_nondim_species_current(solution, 'P', x)

    time = solution.dat['time'][0][0][0]
    phi = solution.dstrbns['phi'][0]
    dx = solution.vectors['dx'][0].flatten()
    dt = np.diff(time)
    mid = x

    dis_param = solution.params['dpt'][0][0][0]
    jd = np.empty(shape=(301))
    jd[0] = None
    for i in range(1, len(time+1)):
        jd[i] = - dis_param / dx[mid] * (
            phi[i,mid+1] - phi[i,mid] - phi[i-1,mid+1] + phi[i-1,mid] )

    pbi = solution.params['pbi'][0][0][0]
    arp = solution.params['ARp'][0][0][0]
    jr = np.empty(shape=(301))
    for i in range(len(time)-1):
        jr[i] = ( pbi - ( solution.dstrbns['phiE'][0][i,0]
                         - solution.dstrbns['phiH'][0][i,-1] ) ) / arp

    return jn, jp, jf

def dimensionalised_current_vector_npP(solution):

    grid = len(solution.vectors['dx'][0].flatten())
    time = len(solution.v)
    for_x_seperate_current = np.empty((3, grid, time))

    for i in range(grid):
        x_current = for_x_seperate_nondim_current_npP(solution, i)
        for j, k in enumerate(x_current):
            for_x_seperate_current[j, i] = k * solution.params['jay'][0][0][0] * 10

    return for_x_seperate_current

def drift_velocity(solution):

    grid = len(solution.vectors['dx'][0].flatten())
    time = len(solution.v)

    current = dimensionalised_current_vector_npP(solution)
    carrier = ['n', 'p', 'P']

    q = solution.params['q'][0][0][0]

    drift_velocity_vector = np.empty((3, grid, time))

    for i, j in enumerate(carrier):
        if j == 'n':
            density = solution.dstrbns[j][0] * solution.params['dE'][0][0][0]
        elif j == 'p':
            density = solution.dstrbns[j][0] * solution.params['dH'][0][0][0]
        elif j == 'P':
            density = solution.dstrbns[j][0] * solution.params['N0'][0][0][0]
        for k, l in enumerate(current[i]):
            drift_velocity_i_k = l / q / density[:, k]
            drift_velocity_vector[i, k] = drift_velocity_i_k

    return drift_velocity_vector

def drift_velocity_set(solutions):

    grid = len(solutions[0].vectors['dx'][0].flatten())
    time = len(solutions[0].v)
    drift_set = np.empty((len(solutions), 3, grid, time))

    for i in range(len(drift_set)):
        drift = drift_velocity(solutions[i])
        drift_set[i] = drift

    return drift_set

def electric_field(solution):

    grid = len(solution.vectors['dx'][0].flatten())
    time = len(solution.v)

    electric_potential = solution.dstrbns['phi'][0] * solution.params['VT'][0][0][0] / 1e3
    dx = solution.vectors['dx'][0].flatten() * solution.widthP / 1e9

    electric_field_sol = - np.diff(electric_potential) / dx

    return electric_field_sol

def electric_field_set(solutions):

    grid = len(solutions[0].vectors['dx'][0].flatten())
    time = len(solutions[0].v)
    field_set = np.empty((len(solutions), grid, time))

    for i in range(len(field_set)):
        field = electric_field(solutions[i]).T
        field_set[i] = field

    return field_set

def plot_scan_tracker_bulk_srh(solution, save=False, titlemod=False):
    fig, ax = plt.subplots()

    ax.plot(range(301), srh_recombination_rate(solution)[:, 200], label='Bulk SRH rate', color='k')
    ax.plot(0, 0, label='Bulk field strength', color='k', linestyle='--')
    ax.plot(0, 0, label='Current density', color='k', linestyle=':')

    ax.axvline(x=100, linewidth=1.5, label='Scan start - 1.2 V', color='C0')
    ax.axvline(x=200, linewidth=1.5, label='Short-circuit - 0 V', color='C3')

    ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    ax.set_xlim(0, 300)
    ax.set_yscale('log')
    ax.set_xlabel('Simulation step count')
    ax.set_ylabel('SRH recombination rate (m$^{-3}$ s$^{-1}$)')

    ax2 = ax.twinx()
    ax2.plot(range(301), electric_field(solution)[:, 200]/1e3, label='Field strength', color='k', linestyle='--')

    ax2.set_ylabel('Field strength (MV m$^{-1}$)')

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.spines["right"].set_visible(True)
    ax3.plot(range(301), solution.j, label='Current density', color='k', linestyle=':')

    ax3.set_ylabel('Current density (mA cm$^{-2}$)')

    # Save file:
    if save is True:
        if titlemod is False:
            fig.savefig(f'scan_tracker_bulk_{int(1200/solution.label)}.png', dpi = 400, bbox_inches='tight')
        else:
            fig.savefig(f'scan_tracker_bulk_{int(1200/solution.label)}_{titlemod}.png', dpi = 400, bbox_inches='tight')

def generation_x(solution, x):

    fph = solution.paramsdic['Fph']
    alpha = solution.paramsdic['alpha']
    b = solution.paramsdic['b']
    inv = solution.paramsdic['inverted']

    if inv == 0:
        l = 1
    elif inv == 1:
        l = -1
    else:
        raise Exception('Direction of light not specified correctly')

    genx = fph * alpha * np.exp(-alpha * (b / 2 + l * (x - b / 2)))

    return genx

def generation_array(solution):

    time = len(solution.j)
    x = solution.vectors['x'][0] * solution.paramsdic['b']
    gen_array = np.empty((time, len(x)))

    for index, depth in enumerate(x):
        gen_array[:, index] = generation_x(solution, depth)

    return gen_array

def srh(n, p, gamma, ni2, tor, tor3):

    srh = gamma*(p-ni2/n)/(1+tor*p/n+tor3/n)*(n>=tor*p)*(n>=tor3) \
        + gamma*(n-ni2/p)/(n/p+tor+tor3/p)*(tor*p>n)*(tor*p>tor3) \
        + gamma*(p*n-ni2)/(n+tor*p+tor3)*(tor3>n)*(tor3>=tor*p)

    return srh

def interface_recombination(solution):

    # ETL surface recombination parameters
    ni2 = solution.paramsdic['ni2']
    kE = solution.paramsdic['kE']
    gammaE = solution.paramsdic['gammaE']
    torE = solution.paramsdic['torE']
    torE3 = solution.paramsdic['torE3']
    nE = solution.dstrbns['nE'][0][:, -1]
    p = solution.dstrbns['p'][0][:, 0]

    etl_recombination = srh(nE, p, gammaE, ni2/kE, torE, torE3)

    # HTL surface recombination parameters
    ni2 = solution.paramsdic['ni2']
    kH = solution.paramsdic['kH']
    gammaH = solution.paramsdic['gammaH']
    torH = solution.paramsdic['torH']
    torH3 = solution.paramsdic['torH3']
    n = solution.dstrbns['n'][0][:, -1]
    pH = solution.dstrbns['pH'][0][:, 0]

    htl_recombination = srh(pH, n, gammaH, ni2/kH, torH, torH3)

    # Rescale to m-2 s-1
    b = solution.paramsdic['b']
    G0 = solution.paramsdic['G0']

    # Convert to 3D
    x = solution.vectors['x'][0]
    xe = solution.vectors['xE'][0]
    xh = solution.vectors['xH'][0]

    etl_psk_gap = abs(xe[-1] - x[0])*b
    htl_psk_gap = abs(xh[0] - x[-1])*b

    return etl_recombination * b * G0 / etl_psk_gap, htl_recombination * b * G0 / htl_psk_gap

def interface_recombination_array(solution):

    # Calculate interface recombination rates
    etl_recomb, htl_recomb = interface_recombination(solution)

    # Define spatial vectors and time
    t_ax = len(solution.paramsdic['time'])

    x = len(solution.vectors['x'][0])
    xe = len(solution.vectors['xE'][0])
    xh = len(solution.vectors['xH'][0])
    x_ax = xe + x + xh

    # Create array
    int_recomb_array = np.zeros((t_ax, x_ax))

    # Insert ETL recombination
    for time, value in enumerate(etl_recomb):
        int_recomb_array[time, xe] = value / 2
        int_recomb_array[time, xe+1] = value / 2

    for time, value in enumerate(htl_recomb):
        int_recomb_array[time, -xh] = value / 2
        int_recomb_array[time, -xh-1] = value / 2

    return int_recomb_array

class AnimationData:

    def __init__(self, label, x, t, j, v, phi, n, p, ndefect, pdefect, g, r, sr):
        self.label = label
        self.x = x
        self.t = t
        self.j = j
        self.v = v
        self.n = n
        self.p = p
        self.ndefect = ndefect
        self.pdefect = pdefect
        self.phi = phi
        self.g = g
        self.r = r
        self.sr = sr

    @classmethod
    def from_ionmonger(cls, sol):

        t_ax = len(sol.paramsdic['time'])
        x_ax = len(sol.vectors['x'][0])
        xe_ax = len(sol.vectors['xE'][0])
        xh_ax = len(sol.vectors['xH'][0])
        nm = 1e-9

        return cls(label = sol.label,
                x = np.concatenate((sol.vectors['xE'][0],
                                      sol.vectors['x'][0],
                                      sol.vectors['xH'][0]),
                                     axis=0) * sol.paramsdic['b'] / nm,
                t = sol.paramsdic['time'] * sol.paramsdic['Tion'],
                j = sol.j,
                v = sol.v,
                n = np.concatenate((sol.dstrbns['nE'][0],
                                    sol.dstrbns['n'][0],
                                    np.zeros((t_ax,
                                             xh_ax))),
                                  axis=1) * sol.dE,
                p = np.concatenate((np.zeros((t_ax,
                                             xe_ax)),
                                    sol.dstrbns['p'][0],
                                    sol.dstrbns['pH'][0]),
                                  axis=1) * sol.dH,
                ndefect = np.concatenate((np.zeros((t_ax,
                                             xe_ax)),
                                          np.full((t_ax, x_ax),
                                                 sol.n0),
                                          np.zeros((t_ax,
                                             xh_ax))),
                                        axis=1),
                pdefect = np.concatenate((np.zeros((t_ax,
                                             xe_ax)),
                                      sol.dstrbns['P'][0],
                                      np.zeros((t_ax,
                                             xh_ax))),
                                    axis=1) * sol.n0,
                phi = np.concatenate((sol.dstrbns['phiE'][0],
                                      sol.dstrbns['phi'][0],
                                      sol.dstrbns['phiH'][0]),
                                     axis=1) * sol.vt,
                g = np.concatenate((np.zeros((t_ax,
                                             xe_ax)),
                                    generation_array(sol),
                                    np.zeros((t_ax,
                                             xh_ax))),
                                    axis=1),
                r = np.concatenate((np.zeros((t_ax,
                                             xe_ax)),
                                    stim.srh_recombination_rate(sol),
                                    np.zeros((t_ax,
                                             xh_ax))),
                                    axis=1),
                sr = interface_recombination_array(sol)
               )

def init_ani(data, ini):

    # Create figure
    fig, ((jv_ax, np_ax), (phi_ax, gr_ax)) = plt.subplots(
        nrows=2, ncols=2, figsize= (10, 10))

    max_scale = 1.1
    min_scale = 0.9

    # JV-time plot
    t_plot = jv_ax.axvline(data.t[ini], c='k')
    v_plot, = jv_ax.plot(data.t[ini], data.v[ini], label='Voltage', c='tab:blue')
    jv_ax2 = jv_ax.twinx()
    j_plot, = jv_ax2.plot(data.t[ini], data.j[ini], c='tab:orange')

    jv_ax.set_xlim(data.t[ini], data.t[-1])
    jv_ax.set_xlabel('Time (s)')

    jv_ax.set_ylim(min(data.v)*min_scale, max(data.v)*max_scale)
    jv_ax.set_ylabel('Applied voltage (V)')

    jv_ax2.set_ylim(0, max(data.j[1:])*max_scale)
    jv_ax2.set_ylabel('Current Density (mA cm$^{-2})$')

    jv_ax.plot(0, 0, label='Current density', c='tab:orange')
    jv_ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand")

    # Densities plot
    n_plot, = np_ax.plot(data.x, data.n[ini, :], label='n', c='tab:blue')
    p_plot, = np_ax.plot(data.x, data.p[ini, :], label='p', c='tab:red')
    ndefect_plot, = np_ax.plot(data.x, data.ndefect[ini, :], label='ndefect', c='tab:orange')
    pdefect_plot, = np_ax.plot(data.x, data.pdefect[ini, :], label='pdefect', c='tab:green')

    np_ax.set_xlim(data.x[0], data.x[-1])
    np_ax.set_xlabel('Depth (nm)')

    dens_array = np.concatenate((data.n, data.p, data.ndefect, data.pdefect))
    np_ax_max = np.amax(dens_array)
    np_ax_min = np.amin(np.where(dens_array > 0, dens_array, float('inf')))

    np_ax.set_ylim(np_ax_min*min_scale, np_ax_max*max_scale)
    np_ax.set_yscale('log')
    np_ax.set_ylabel('Number density (m$^{-3}$)')

    np_ax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)

    # Electric potential plot
    phi_plot, = phi_ax.plot(data.x, data.phi[ini, :], c='tab:purple')

    phi_ax.set_xlim(data.x[0], data.x[-1])
    phi_ax.set_xlabel('Depth (nm)')

    phi_ax.set_ylim(np.amin(data.phi)*max_scale, np.amax(data.phi)*max_scale)
    phi_ax.set_ylabel('Electric potential (V)')

    # Generation-recombination plot
    g_plot, = gr_ax.plot(data.x, data.g[ini, :], label='Generation')
    r_plot, = gr_ax.plot(data.x, data.r[ini, :], label='SRH recombination')
    sr_plot, = gr_ax.plot(data.x, data.sr[ini, :], label='Interface Recombination')

    gr_ax.set_xlim(data.x[0], data.x[-1])
    gr_ax.set_xlabel('Depth (nm)')

    rate_array = np.concatenate((data.g, data.r, data.sr))
    gr_ax_max = np.amax(rate_array)
    gr_ax_min = np.amin(np.where(rate_array > 0, rate_array, float('inf')))

    gr_ax.set_ylim(gr_ax_min*min_scale, gr_ax_max*max_scale)
    gr_ax.set_yscale('log')
    gr_ax.set_ylabel('Rate (m$^{-3}$s$^{-1}$)')

    gr_ax.legend(bbox_to_anchor=(0, -0.32, 1, 0), loc="lower left", mode="expand")

    # Make plot pretty
    fig.tight_layout()

    return fig, ini, t_plot, v_plot, j_plot, n_plot, p_plot, phi_plot,\
        ndefect_plot, pdefect_plot, g_plot, r_plot, sr_plot

def update_ani(num, ini, t_plot, v_plot, j_plot, n_plot, p_plot, phi_plot,
               ndefect_plot, pdefect_plot, g_plot, r_plot, sr_plot, data):

    t_plot.set_xdata(data.t[num+ini])
    v_plot.set_data(data.t[:num+ini], data.v[:num+ini])
    j_plot.set_data(data.t[:num+ini], data.j[:num+ini])
    n_plot.set_ydata(data.n[num+ini, :])
    p_plot.set_ydata(data.p[num+ini, :])
    phi_plot.set_ydata(data.phi[num+ini, :])
    ndefect_plot.set_ydata(data.ndefect[num+ini, :])
    pdefect_plot.set_ydata(data.pdefect[num+ini, :])
    g_plot.set_ydata(data.g[num+ini, :])
    r_plot.set_ydata(data.r[num+ini, :])
    sr_plot.set_ydata(data.sr[num+ini, :])

    return ini, t_plot, v_plot, j_plot, n_plot, p_plot, phi_plot, ndefect_plot, pdefect_plot, g_plot, r_plot, sr_plot

def animate_solution(file, output, ini, length):
    sol = stim.Solution.from_single(file, key='sol')
    data = AnimationData.from_ionmonger(sol)

    init_fig, ini_ini, t_plot, init_v_plot, init_j_plot, init_n_plot, init_p_plot, phi_plot,\
        init_ndefect_plot, init_pdefect_plot, init_g_plot, init_r_plot, init_sr_plot = init_ani(data, ini(data))

    ani = animation.FuncAnimation(init_fig,
                                  update_ani,
                                  fargs=(ini_ini,
                                         t_plot,
                                         init_v_plot,
                                         init_j_plot,
                                         init_n_plot,
                                         init_p_plot,
                                         phi_plot,
                                         init_ndefect_plot,
                                         init_pdefect_plot,
                                         init_g_plot,
                                         init_r_plot,
                                         init_sr_plot,
                                         data
                                        ),
                                  frames=length(data),
                                  interval=100
                                 )

    ani.save(f'{output}.mp4', dpi=500)
