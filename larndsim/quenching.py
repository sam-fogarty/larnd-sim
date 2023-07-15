"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
import larnestpy
import numpy as np
from .consts import detector, physics, light

def quench(track, calc, tracks_dtype):
    """
    This function takes as input an array of track segments and calculates
    the number of electrons and photons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        track (:obj:`numpy.ndarray`): array containing the track segment information
    """
    track = np.array(track, dtype=tracks_dtype)
    dEdx = track['dEdx']
    dE = track['dE']
    dx = track['dx']
    recomb = 0
    pdg_id = track['pdgId']
    E_cut = 1.0 # MeV
    
    if pdg_id == 11:
        m_electron = 0.510999
        p_start = np.sqrt(np.sum(track['traj_pxyz_start']**2))
        E = np.sqrt(m_electron**2 + p_start**2) - m_electron
        if E < E_cut:
            # use LArNEST ER model
            result = calc.full_calculation(larnestpy.LArInteraction.ER, E*1e3, dx, detector.E_FIELD*1e3, detector.LAR_DENSITY, False)
            recomb = result.yields.Ne / (result.yields.Ne + result.yields.Nph)
    elif pdg_id == 1000020040:
        # use LArNEST alpha model
        m_alpha = 3753.63
        p_start = np.sqrt(np.sum(track['traj_pxyz_start']**2))
        E = np.sqrt(m_alpha**2 + p_start**2) - m_alpha
        result = calc.get_alpha_yields(E*1e3, detector.E_FIELD*1e3, detector.LAR_DENSITY)
        recomb = result.Ne / (result.Ne + result.Nph)
    elif pdg_id == 1000180400:
        # use LArNEST NR model for recoiling Ar40
        m_Ar40 = 37284.0
        p_start = np.sqrt(np.sum(track['traj_pxyz_start']**2))
        E = np.sqrt(m_Ar40**2 + p_start**2) - m_Ar40
        result = calc.get_nr_yields(E*1e3, detector.E_FIELD*1e3, detector.LAR_DENSITY)
        recomb = result.Ne / (result.Ne + result.Nph)
    else:
        # use LArNEST dE/dx model
        result = calc.get_dEdx_recombination_probability(dEdx, detector.E_FIELD*1e3)
        recomb = result
    if isnan(recomb):
        raise RuntimeError("Invalid recombination value")
    
    track['n_electrons'] = recomb * dE / physics.W_ION
    track['n_photons'] = (dE/light.W_PH - track['n_electrons']) * light.SCINT_PRESCALE
    
    return track
