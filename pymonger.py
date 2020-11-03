import numpy as np
import scipy.io as sio

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
        self.revarea = np.trapz(self.revy, x=-self.v[self.RevVoc:self.Jsc+1])

        self.fwdy = self.j[self.Jsc:self.FwdVoc+1]
        self.fwdarea = np.trapz(self.fwdy, x=self.v[self.Jsc:self.FwdVoc+1])

        self.degreehyst = ((self.revarea - self.fwdarea) / self.revarea) * 100
