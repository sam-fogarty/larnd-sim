"""
Sets ligth-related constants
"""
import yaml
import numpy as np
import os
import numbers

from . import detector

LIGHT_SIMULATED = True

ENABLE_LUT_SMEARING = False

N_OP_CHANNEL = 0
OP_CHANNEL_EFFICIENCY = np.ones(0)
OP_CHANNEL_TO_TPC = np.zeros(0)
TPC_TO_OP_CHANNEL = np.zeros((0,0))

#: Prescale factor analogous to ScintPreScale in LArSoft FIXME
SCINT_PRESCALE = 1
#: Ion + excitation work function in `MeV`
W_PH = 19.5e-6 # MeV

#: Step size for light simulation [microseconds]
LIGHT_TICK_SIZE = 0.001 # us
#: Pre- and post-window for light simulation [microseconds]
LIGHT_WINDOW = (1, 10) # us

#: Fraction of total light emitted from singlet state
SINGLET_FRACTION = 0.3
#: Singlet decay time [microseconds]
TAU_S = 0.001 # us
#: Triplet decay time [microseconds]
TAU_T = 1.530

#: Conversion from PE/microsecond to ADC
DEFAULT_LIGHT_GAIN = -2.30 # ADC * us/PE
LIGHT_GAIN = np.zeros((0,))
#: Set response model type (0=RLC response, 1=arbitrary input)
SIPM_RESPONSE_MODEL = 0
#: Response RC time [microseconds]
LIGHT_RESPONSE_TIME = 0.055
#: Reponse oscillation period [microseconds]
LIGHT_OSCILLATION_PERIOD = 0.095
#: Sample rate for input noise spectrum [microseconds]
LIGHT_DET_NOISE_SAMPLE_SPACING = 0.01 # us
#: Arbitrary input model (normalized to sum of 1)
IMPULSE_MODEL = np.array([1,0])
#: Arbitrary input model tick size [microseconds]
IMPULSE_TICK_SIZE = 0.001

#: Number of SiPMs per detector (used by trigger)
OP_CHANNEL_PER_TRIG = 6
#: Light trigger mode (0: threshold each module, 1: beam and threshold)
LIGHT_TRIG_MODE = 0
#: Total detector light threshold [ADC] (one value for every OP_CHANNEL_PER_TRIG detector sum)
LIGHT_TRIG_THRESHOLD = np.zeros((0,))
#: Light digitization window [microseconds]
LIGHT_TRIG_WINDOW = (0.9, 1.66) # us
#: Light waveform sample rate [microseconds]
LIGHT_DIGIT_SAMPLE_SPACING = 0.01 # us
#: Light digitizer bits
LIGHT_NBIT = 10

def set_light_properties(detprop_file):
    """
    The function loads the detector properties YAML file
    and stores the light-related constants as global variables

    Args:
        detprop_file (str): detector properties YAML filename

    """
    global LIGHT_SIMULATED

    global N_OP_CHANNEL
    global OP_CHANNEL_EFFICIENCY
    global OP_CHANNEL_TO_TPC
    global TPC_TO_OP_CHANNEL

    global ENABLE_LUT_SMEARING
    global LIGHT_TICK_SIZE
    global LIGHT_WINDOW

    global SINGLET_FRACTION
    global TAU_S
    global TAU_T

    global LIGHT_GAIN
    global SIPM_RESPONSE_MODEL
    global LIGHT_RESPONSE_TIME
    global LIGHT_OSCILLATION_PERIOD
    global LIGHT_DET_NOISE_SAMPLE_SPACING
    global IMPULSE_MODEL
    global IMPULSE_TICK_SIZE

    global OP_CHANNEL_PER_TRIG
    global LIGHT_TRIG_MODE
    global LIGHT_TRIG_THRESHOLD
    global LIGHT_TRIG_WINDOW
    global LIGHT_DIGIT_SAMPLE_SPACING
    global LIGHT_NBIT

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        LIGHT_SIMULATED = bool(detprop.get('light_simulated', LIGHT_SIMULATED))

        mod_ids = detector.get_n_modules(detprop_file)
        n_tpc = len(mod_ids)*2
        N_OP_CHANNEL = detprop['n_op_channel']
        if N_OP_CHANNEL % n_tpc != 0:
            raise ValueError("N_OP_CHANNEL should be a multiple of n_tpc.")
        if N_OP_CHANNEL % OP_CHANNEL_PER_TRIG != 0:
            raise ValueError("N_OP_CHANNEL should be a multiple of number of SiPM per light unit (The default is 6).")
        OP_CHANNEL_EFFICIENCY = np.array(detprop.get('op_channel_efficiency', OP_CHANNEL_EFFICIENCY))
        if OP_CHANNEL_EFFICIENCY.size == 1:
            OP_CHANNEL_EFFICIENCY = np.full(N_OP_CHANNEL, OP_CHANNEL_EFFICIENCY)

        try:
            tpc_to_op_channel = detprop['tpc_to_op_channel']
            OP_CHANNEL_TO_TPC = np.zeros((N_OP_CHANNEL,), int)
            TPC_TO_OP_CHANNEL = np.zeros((len(tpc_to_op_channel), len(tpc_to_op_channel[0])), int)
            for itpc in range(len(tpc_to_op_channel)):
                TPC_TO_OP_CHANNEL[itpc] = np.array(tpc_to_op_channel[itpc])
                for idet in tpc_to_op_channel[itpc]:
                    OP_CHANNEL_TO_TPC[idet] = itpc
        except:
            n_op_per_tpc = int(N_OP_CHANNEL/n_tpc)
            OP_CHANNEL_TO_TPC = np.zeros((N_OP_CHANNEL,), int)
            TPC_TO_OP_CHANNEL = np.zeros((n_tpc, n_op_per_tpc), int)
            for itpc in range(n_tpc):
                TPC_TO_OP_CHANNEL[itpc] = np.arange(itpc*n_op_per_tpc, (itpc+1)*n_op_per_tpc)
                for idet in TPC_TO_OP_CHANNEL[itpc]:
                    OP_CHANNEL_TO_TPC[idet] = itpc

        ENABLE_LUT_SMEARING = bool(detprop.get('enable_lut_smearing', ENABLE_LUT_SMEARING))
        LIGHT_TICK_SIZE = float(detprop.get('light_tick_size', LIGHT_TICK_SIZE))
        LIGHT_WINDOW = tuple(detprop.get('light_window', LIGHT_WINDOW))
        assert len(LIGHT_WINDOW) == 2

        SINGLET_FRACTION = float(detprop.get('singlet_fraction', SINGLET_FRACTION))
        TAU_S = float(detprop.get('tau_s', TAU_S))
        TAU_T = float(detprop.get('tau_t', TAU_T))

        LIGHT_GAIN = np.array(detprop.get('light_gain', [DEFAULT_LIGHT_GAIN]))
        if LIGHT_GAIN.size == 1:
            LIGHT_GAIN = np.full(N_OP_CHANNEL, LIGHT_GAIN)
        assert LIGHT_GAIN.shape == OP_CHANNEL_EFFICIENCY.shape
        SIPM_RESPONSE_MODEL = int(detprop.get('sipm_response_model', SIPM_RESPONSE_MODEL))
        assert SIPM_RESPONSE_MODEL in (0,1)
        LIGHT_DET_NOISE_SAMPLE_SPACING = float(detprop.get('light_det_noise_sample_spacing', LIGHT_DET_NOISE_SAMPLE_SPACING))
        LIGHT_RESPONSE_TIME = float(detprop.get('light_response_time', LIGHT_RESPONSE_TIME))
        LIGHT_OSCILLATION_PERIOD = float(detprop.get('light_oscillation_period', LIGHT_OSCILLATION_PERIOD))
        impulse_model_filename = str(detprop.get('impulse_model', ''))
        if impulse_model_filename and SIPM_RESPONSE_MODEL == 1:
            try:
                # first try to load from current directory
                IMPULSE_MODEL = np.load(impulse_model_filename)
            except FileNotFoundError:
                # then try from larnd-sim base directory
                try:
                    IMPULSE_MODEL = np.load(os.path.join(os.path.dirname(__file__), '../../') + impulse_model_filename)
                except FileNotFoundError:
                    SIPM_RESPONSE_MODEL = 0
                    print("Impulse model file not found:", impulse_model_filename, ", and setting SIPM_RESPONSE_MODEL to 0 (RLC model).")
        IMPULSE_TICK_SIZE = float(detprop.get('impulse_tick_size', IMPULSE_TICK_SIZE))

        OP_CHANNEL_PER_TRIG = int(detprop.get('op_channel_per_det', OP_CHANNEL_PER_TRIG))
        LIGHT_TRIG_MODE = int(detprop.get('light_trig_mode', LIGHT_TRIG_MODE))
        assert LIGHT_TRIG_MODE in (0,1)
        # One threshold for an ArCLight unit or three LCM units (6 SiPMs each)
        # A single threshold for all channels
        if isinstance(detprop['light_trig_threshold'], (float, int)):
            LIGHT_TRIG_THRESHOLD = np.full(N_OP_CHANNEL // OP_CHANNEL_PER_TRIG, float(detprop['light_trig_threshold']))
        # Assuming the same threshold is applied for all ACL and another one for all LCM
        elif isinstance(detprop['light_trig_threshold'], list) and len(detprop['light_trig_threshold']) == 2:
            LIGHT_TRIG_THRESHOLD = np.tile(np.array(detprop['light_trig_threshold'], dtype=float), N_OP_CHANNEL // OP_CHANNEL_PER_TRIG)
        else:
            LIGHT_TRIG_THRESHOLD = np.array(detprop['light_trig_threshold'], dtype=float)
            if len(LIGHT_TRIG_THRESHOLD) != (N_OP_CHANNEL // OP_CHANNEL_PER_TRIG):
                raise ValueError("The light_trig_threshold is provided as a list but with a length not matched with n_op_channel.")
        LIGHT_TRIG_WINDOW = tuple(detprop.get('light_trig_window', LIGHT_TRIG_WINDOW))
        assert len(LIGHT_TRIG_WINDOW) == 2
        LIGHT_DIGIT_SAMPLE_SPACING = float(detprop.get('light_digit_sample_spacing', LIGHT_DIGIT_SAMPLE_SPACING))
        LIGHT_NBIT = int(detprop.get('light_nbit', LIGHT_NBIT))



    except KeyError:
        LIGHT_SIMULATED = False
        LIGHT_TRIG_MODE = int(detprop.get('light_trig_mode', LIGHT_TRIG_MODE))
        assert LIGHT_TRIG_MODE in (0,1)
