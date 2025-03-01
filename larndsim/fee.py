"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
import cupy as cp
import h5py
import yaml
import warnings

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32
from math import exp, floor

from larpix.packet import Packet_v2, TimestampPacket, TriggerPacket, SyncPacket, PacketCollection
from larpix.key import Key
from larpix.format import hdf5format

from .pixels_from_track import id2pixel

from .consts.units import mV, e
from .consts import units, detector, light, sim

import logging
logging.basicConfig()
logger = logging.getLogger('fee')
logger.setLevel(logging.WARNING)
logger.info("ELECTRONICS SIMULATION")

def get_trig_io():
    """
    Returns the io_group if the trigger is only forwarded to one pacman
    """
    if light.LIGHT_TRIG_MODE == 0: #: Threshold/LRS trigger
        trig_io = 2
    elif light.LIGHT_TRIG_MODE == 1: #: Beam trigger
        trig_io = 1
    return trig_io

def rotate_tile(pixel_id, tile_id):
    """
    Returns the pixel ID of the rotated tile.

    Args:
        pixel_id(int): pixel ID
        tile_id(int): tile ID

    Returns:
        tuple: pixel indeces
    """
    axes = detector.TILE_ORIENTATIONS[tile_id]
    x_axis = axes[2]
    y_axis = axes[1]

    pix_x = pixel_id[0]
    if x_axis < 0:
        pix_x = detector.N_PIXELS_PER_TILE[0]-pixel_id[0]-1

    pix_y = pixel_id[1]
    if y_axis < 0:
        pix_y = detector.N_PIXELS_PER_TILE[1]-pixel_id[1]-1

    return pix_x, pix_y


def gen_event_times(nevents, t0=detector.NON_BEAM_EVENT_GAP):
    """
    Generate sequential event times assuming events are uncorrelated

    Args:
        nevents(int): number of event times to generate
        t0(int): offset to apply [microseconds]

    Returns:
        array: shape `(nevents,)`, sequential event times [microseconds]
    """
    event_start_time = cp.random.exponential(scale=detector.EVENT_RATE, size=int(nevents))
    event_start_time = cp.cumsum(event_start_time)
    event_start_time += t0

    return event_start_time


def export_to_hdf5(event_id_list,
                   adc_list,
                   adc_ticks_list,
                   unique_pix,
                   current_fractions,
                   track_ids,
                   traj_ids,
                   filename,
                   event_start_times,
                   light_trigger_times=None,
                   light_trigger_event_id=None,
                   light_trigger_modules=None,
                   bad_channels=None,
                   i_mod=-1):
    """
    Saves the ADC counts in the LArPix HDF5 format.
    Args:
        event_id_list (:obj:`numpy.ndarray`): event ids for each tick;
                shape (nticks, max_adcs); dtype uint32
        adc_list (:obj:`numpy.ndarray`): ADC values for each tick;
                shape (nticks, max_adcs); dtype float64
        adc_ticks_list (:obj:`numpy.ndarray`): timestamps for each tick;
                shape (nticks, max_adcs); dtype float64
        unique_pix (:obj:`numpy.ndarray`): pixel IDs for each tick;
                shape (nticks,); dtype int32
        current_fractions (:obj:`numpy.ndarray`): fractional current induced by
            each track on each pixel;
                shape (nticks, max_adcs, max_backtracks); dtype float64
        track_ids (:obj:`numpy.ndarray`): track IDs associated to each pixel;
                shape (nticks, max_backtracks); dtype int64
        filename (str): filename of HDF5 output file
        event_start_times (:obj:`numpy.ndarray`): timestamps of start of each
            unique event [in microseconds];
                shape (nevents,); dtype float64
        light_trigger_times (:obj:`numpy.ndarray`): light trigger timestamps
            (relative to event t0) [in microseconds];
                shape (ntrigs,); dtype float64
        light_trigger_event_id (:obj:`numpy.ndarray`): event id for each light trigger;
                shape (ntrigs,); dtype uint32
        light_trigger_modules (:obj:`numpy.ndarray`): module id for each light trigger;
                shape (ntrigs,); dtype int64
        bad_channels (dict): dictionary mapping a chip key to a list of bad channels
        i_mod (int): module index for saving the result in each module
            individually if needed.
    Returns:
        tuple: a tuple containing the list of LArPix packets and the list of
            entries for the `mc_packets_assn` dataset
    """

    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = io_groups if i_mod < 0 else io_groups[(i_mod-1)*2: i_mod*2]
    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_mc_trj = []
    packets_frac = []

    packets_mc_ds = []
    last_event = -1

    if bad_channels:
        with open(bad_channels, 'r') as bad_channels_file:
            bad_channels_list = yaml.load(bad_channels_file, Loader=yaml.FullLoader)

    unique_events, unique_events_inv = np.unique(event_id_list[...,0], return_inverse=True)
    event_start_time_list = (event_start_times[unique_events_inv] / detector.CLOCK_CYCLE).astype(int)
    light_trigger_times = np.empty((0,)) if light_trigger_times is None else light_trigger_times
    light_trigger_event_id = np.empty((0,), dtype=int) if light_trigger_event_id is None else light_trigger_event_id

    rollover_count = 0
    last_time_tick = -1
    for itick, adcs in enumerate(adc_list):
        ts = adc_ticks_list[itick]
        pixel_id = unique_pix[itick]

        pix_x, pix_y, plane_id = id2pixel(pixel_id)
        module_id = plane_id//2+1

        if module_id not in detector.MODULE_TO_IO_GROUPS.keys():
            logger.warning("Pixel ID not valid %i" % module_id)
            continue

        tile_x = int(pix_x//detector.N_PIXELS_PER_TILE[0])
        tile_y = int(pix_y//detector.N_PIXELS_PER_TILE[1])
        anode_id = 0 if plane_id % 2 == 0 else 1
        tile_id = detector.TILE_MAP[anode_id][tile_x][tile_y]

        for iadc, adc in enumerate(adcs):
            t = ts[iadc]

            if adc > digitize(0):
                while True:
                    event = event_id_list[itick,iadc]
                    event_t0 = event_start_time_list[itick]
                    time_tick = int(np.floor(t / detector.CLOCK_CYCLE + event_t0))

                    if event_t0 > detector.CLOCK_RESET_PERIOD-1 or time_tick > detector.CLOCK_RESET_PERIOD-1:
                        # rollover (reset) at either PPS or at the 31-bit clock limit
                        rollover_count += 1
                        # FIXME disable this sync packets fill as it may overlap with what is already filled
                        #for io_group in io_groups:
                        #    packets.append(SyncPacket(sync_type=b'S',
                        #                              timestamp=detector.CLOCK_RESET_PERIOD-1, io_group=io_group))
                        #    packets_mc_evt.append([-1])
                        #    packets_mc_trk.append([-1] * track_ids.shape[1])
                        #    packets_frac.append([0] * current_fractions.shape[2])
                        event_start_time_list[itick:] -= detector.CLOCK_RESET_PERIOD
                    else:
                        break
                
                event_t0 = event_t0 % detector.CLOCK_RESET_PERIOD
                time_tick = time_tick % detector.CLOCK_RESET_PERIOD

                # FIXME light.LIGHT_TRIG_MODE != 0 should also be here
                # This trigger packet block should only be activated with trigger forwarding scheme to individual modules
                if light.LIGHT_TRIG_MODE != 1:
                    # new event, insert light triggers and timestamp flag
                    if event != last_event:
                        for io_group in io_groups:
                            packets.append(TimestampPacket(timestamp=event_start_times[unique_events_inv[itick]] * units.mus / units.s))
                            packets[-1].chip_key = Key(io_group,0,0)

                            packets_mc_evt.append([-1])
                            packets_mc_trk.append([-1] * track_ids.shape[1])
                            packets_mc_trj.append([-1] * traj_ids.shape[1])
                            packets_frac.append([0] * current_fractions.shape[2])

                            packets.append(SyncPacket(sync_type=b'S', timestamp=time_tick , io_group=io_group))
                            packets_mc_evt.append([-1])
                            packets_mc_trk.append([-1] * track_ids.shape[1])
                            packets_mc_trj.append([-1] * traj_ids.shape[1])
                            packets_frac.append([0] * current_fractions.shape[2])

                        trig_mask = light_trigger_event_id == event
                        if any(trig_mask):
                            for t_trig, module_trig in zip(light_trigger_times[trig_mask], light_trigger_modules[trig_mask]):
                                t_trig = int(np.floor(t_trig / detector.CLOCK_CYCLE + event_t0)) % detector.CLOCK_RESET_PERIOD
                                if light.LIGHT_TRIG_MODE == 0:
                                    for io_group in detector.MODULE_TO_IO_GROUPS[int(module_trig)]:
                                        packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                        packets_mc_evt.append([-1])
                                        packets_mc_trk.append([-1] * track_ids.shape[1])
                                        packets_mc_trj.append([-1] * traj_ids.shape[1])
                                        packets_frac.append([0] * current_fractions.shape[2])
                                # redundant here
                                elif light.LIGHT_TRIG_MODE == 1:
                                    if module_trig == 1 or module_trig == 0: #1, beam trigger; 2, threshold trigger
                                        io_group = get_trig_io()
                                    packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                    packets_mc_evt.append([-1])
                                    packets_mc_trk.append([-1] * track_ids.shape[1])
                                    packets_mc_trj.append([-1] * traj_ids.shape[1])
                                    packets_frac.append([0] * current_fractions.shape[2])
                        last_event = event

                p = Packet_v2()

                try:
                    chip, channel = detector.PIXEL_CONNECTION_DICT[rotate_tile((pix_x % detector.N_PIXELS_PER_TILE[0],
                                                                                pix_y % detector.N_PIXELS_PER_TILE[1]),
                                                                   tile_id)]
                except KeyError:
                    logger.warning(f"Pixel ID not valid {pixel_id}")
                    continue

                p.dataword = int(adc)
                p.timestamp = time_tick

                try:
                    io_group_io_channel = detector.TILE_CHIP_TO_IO[tile_id][chip]
                except KeyError:
                    logger.warning(f"Chip {chip} on tile {tile_id} not found")
                    continue

                io_group, io_channel = io_group_io_channel // 1000, io_group_io_channel % 1000
                io_group = detector.MODULE_TO_IO_GROUPS[module_id][io_group-1]
                chip_key = "%i-%i-%i" % (io_group, io_channel, chip)

                if bad_channels:
                    if chip_key in bad_channels_list:
                        if channel in bad_channels_list[chip_key]:
                            logger.info(f"Channel {channel} on chip {chip_key} disabled")
                            continue
                p.chip_key = chip_key
                p.channel_id = channel
                p.receipt_timestamp = time_tick
                p.packet_type = 0
                p.first_packet = 1
                p.assign_parity()

                if not time_tick==last_time_tick:
                    # timestamp packet every time there is a new "message"
                    # the logic in real data for when a timestamp packet is complicated and depends on pacman CPU speed, packet creation rate
                    # best simple approximation is that any group of packets with the same timestamp get a single timestamp packet
                    last_time_tick = time_tick
                    packets.append(TimestampPacket(timestamp=np.floor(event_start_time_list[0] * detector.CLOCK_CYCLE * units.mus/units.s)) ) # s
                    packets[-1].chip_key = Key(io_group,0,0)

                    packets_mc_evt.append([-1])
                    packets_mc_trk.append([-1] * sim.MAX_TRACKS_PER_PIXEL)
                    packets_mc_trj.append([-1] * sim.MAX_TRACKS_PER_PIXEL)
                    packets_frac.append([0] * sim.MAX_TRACKS_PER_PIXEL)

                packets_mc_evt.append([event])
                packets_mc_trk.append(track_ids[itick])
                packets_mc_trj.append(traj_ids[itick])
                packets_frac.append(current_fractions[itick][iadc])
                packets.append(p)
                
            else:
                break

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1)
        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])

        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        # First, sort the back-tracking information by the magnitude of the fraction
        packets_frac = np.array(packets_frac)
        packets_mc_trk = np.array(packets_mc_trk)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_mc_evt = np.array(packets_mc_evt)
        
        frac_order = np.flip(np.argsort(packets_frac,axis=1),axis=1)
        ass_segment_ids = np.take_along_axis(packets_mc_trk,   frac_order, axis=1)
        ass_trajectory_ids = np.take_along_axis(packets_mc_trj, frac_order, axis=1)
        ass_fractions = np.take_along_axis(packets_frac, frac_order, axis=1)

        # Second, only store the relevant portion.
        if ass_segment_ids.shape[1] >= sim.ASSOCIATION_COUNT_TO_STORE:
            packets_mc_ds['segment_ids'] = ass_segment_ids[:,:sim.ASSOCIATION_COUNT_TO_STORE]
            packets_mc_ds['fraction' ] = ass_fractions[:,:sim.ASSOCIATION_COUNT_TO_STORE]
        else:
            num_to_pad = sim.ASSOCIATION_COUNT_TO_STORE - ass_segment_ids.shape[1]
            packets_mc_ds['segment_ids'] = np.pad(ass_segment_ids,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=-1)
            packets_mc_ds['fraction' ] = np.pad(ass_fractions,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=0.)


        ass_track_ids = np.full(ass_trajectory_ids.shape,fill_value=-1,dtype=np.int32)
        ass_fractions_track = np.full(ass_fractions.shape,fill_value=0.,dtype=np.float32)
        for pidx, tids in enumerate(ass_trajectory_ids):
            mask = tids > -1
            for tidx, unique_tid in enumerate(np.unique(tids[mask])):
                ass_track_ids[pidx][tidx] = unique_tid
                ass_fractions_track[pidx][tidx] = np.sum(ass_fractions[pidx][mask][tids[mask]==unique_tid])

        if ass_segment_ids.shape[1] >= sim.ASSOCIATION_COUNT_TO_STORE:
            packets_mc_ds['file_traj_ids'] = ass_track_ids[:,:sim.ASSOCIATION_COUNT_TO_STORE]
            packets_mc_ds['fraction_traj'] = ass_fractions_track[:,:sim.ASSOCIATION_COUNT_TO_STORE]
        else:
            num_to_pad = sim.ASSOCIATION_COUNT_TO_STORE - ass_track_ids.shape[1]
            packets_mc_ds['file_traj_ids'] = np.pad(ass_track_ids,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=-1)
            packets_mc_ds['fraction_traj' ] = np.pad(ass_fractions_track,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=0.)

        packets_mc_ds['event_ids'] = packets_mc_evt

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,))
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

            f['configs'].attrs['vdrift'] = detector.V_DRIFT
            f['configs'].attrs['long_diff'] = detector.LONG_DIFF
            f['configs'].attrs['tran_diff'] = detector.TRAN_DIFF
            f['configs'].attrs['lifetime'] = detector.ELECTRON_LIFETIME
            f['configs'].attrs['drift_length'] = detector.DRIFT_LENGTH

    return packets, packets_mc_ds

def export_sync_to_hdf5(filename, sync_times, i_mod=-1):
    """
    Saves sync packets in the LArPix HDF5 format.
    Args:
        sync_times (:obj:`numpy.ndarray`): list of sync timestamps [us]
    Returns:
        tuple: a tuple containing the list of LArPix sync packets and the list of entries for the `mc_packets_assn` dataset
    """
    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = detector.MODULE_TO_IO_GROUPS[i_mod] if i_mod > 0 else io_groups

    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_frac = []
    packets_mc_trj = []
    packets_frac_trj =[]

    sync_ticks = sync_times / detector.CLOCK_CYCLE # us -> time tick
    for sync_tick in sync_ticks:
        if sync_tick % detector.CLOCK_RESET_PERIOD != 0:
            warnings.warn("The provided sync time is not the mutiply of the reset period!")
            sync_tick = sync_tick // detector.CLOCK_RESET_PERIOD * detector.CLOCK_RESET_PERIOD
        for io_group in io_groups:
            packets.append(SyncPacket(sync_type=b'S', timestamp=sync_tick, io_group=io_group))
            packets_mc_evt.append(np.array([-1]))

            packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1)

        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])

        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        packets_frac = np.array(packets_frac)
        packets_mc_trk   = np.array(packets_mc_trk)
        packets_mc_evt   = np.array(packets_mc_evt)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_frac_trj = np.array(packets_frac_trj)

        packets_mc_ds['event_ids'] = packets_mc_evt
        packets_mc_ds['segment_ids'] = packets_mc_trk
        packets_mc_ds['fraction' ] = packets_frac
        packets_mc_ds['file_traj_ids'] = packets_mc_trj
        packets_mc_ds['fraction_traj'] = packets_frac_trj

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,))
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

    return packets, packets_mc_ds

def export_timestamp_trigger_to_hdf5(filename, event_start_times, i_mod=-1):
    """
    Saves timestamp and trigger packets in the LArPix HDF5 format.
    Args:
        event_start_times (:obj:`numpy.ndarray`): list of timestamps for start each unique event [in microseconds]
    Returns:
        tuple: a tuple containing the list of LArPix timestamp and trigger packets and the list of entries for the `mc_packets_assn` dataset
    """
    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = detector.MODULE_TO_IO_GROUPS[i_mod] if i_mod > 0 else io_groups

    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_frac = []
    packets_mc_trj = []
    packets_frac_trj =[]

    for evt_time in event_start_times:

        t_trig = int(np.floor(evt_time / detector.CLOCK_CYCLE)) % detector.CLOCK_RESET_PERIOD # tick

        io_group = get_trig_io()

        # timestamp packets
        packets.append(TimestampPacket(timestamp=evt_time*units.mus/units.s)) # s
        packets[-1].chip_key = Key(io_group,0,0)
        packets_mc_evt.append(np.array([-1]))
        packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

        # trigger packets
        packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig)) # tick
        packets_mc_evt.append(np.array([-1]))
        packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1)

        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])
        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        packets_frac = np.array(packets_frac)
        packets_mc_trk   = np.array(packets_mc_trk)
        packets_mc_evt   = np.array(packets_mc_evt)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_frac_trj = np.array(packets_frac_trj)

        packets_mc_ds['event_ids'] = packets_mc_evt
        packets_mc_ds['segment_ids'] = packets_mc_trk
        packets_mc_ds['fraction' ] = packets_frac
        packets_mc_ds['file_traj_ids'] = packets_mc_trj
        packets_mc_ds['fraction_traj'] = packets_frac_trj

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,))
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

    return packets, packets_mc_ds

def digitize(integral_list, gain=detector.GAIN * mV / e):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list(:obj:`numpy.ndarray`): list of charge collected by each pixel
        gain(:obj:`numpy.ndarray`): list of gain values (or float) for each pixel

    Returns:
        :obj:`numpy.ndarray`: list of ADC values for each pixel
    """
    xp = cp.get_array_module(integral_list)
    adcs = xp.floor(xp.minimum(xp.maximum((integral_list * gain + detector.V_PEDESTAL * mV - detector.V_CM * mV), 0)
                                * detector.ADC_COUNTS / (detector.V_REF * mV - detector.V_CM * mV), detector.ADC_COUNTS-1))

    return adcs

@cuda.jit
def get_adc_values(pixels_signals,
                   pixels_signals_tracks,
                   num_backtrack,
                   offset_backtrack,
                   time_ticks,
                   adc_list,
                   adc_ticks_list,
                   time_padding,
                   rng_states,
                   current_fractions,
                   pixel_thresholds):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel. Shape (n_unique_pix, n_ticks).
        pixels_signals_tracks (:obj:`numpy.ndarray`): list of induced currents
            for each track that induces current on each pixel.
            Jagged; "shape" (n_unique_pix, n_ticks, num_backtrack[ipix]).
        num_backtrack (:obj:`numpy.ndarray`): For a given pixel, the number of
            backtracked track segments. Shape (n_unique_pix,).
        offset_backtrack (:obj:`numpy.ndarray`): For a given pixel, the offset
            into `pixels_signals_tracks` for that pixel's data.
            Shape (n_unique_pix,).
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel.
            Shape (n_ticks+1,).
        adc_list (:obj:`numpy.ndarray`): Output; list of integrated charges for each
            pixel. Shape (n_unique_pix, max_adc_vals).
        adc_ticks_list (:obj:`numpy.ndarray`): Output; list of the time ticks that
            correspond to each integrated charge. Shape (n_unique_pix, max_adc_vals).
        time_padding (float): time interval to add to each time tick.
        rng_states (:obj:`numpy.ndarray`): array of random states for noise
            generation. Shape (TPB*BPG,).
        current_fractions (:obj:`numpy.ndarray`): Output; 2D array that will contain
            the fraction of current induced on the pixel by each track.
            Shape (n_unique_pix, max_adc_vals, MAX_TRACKS_PER_PIXEL).
            TODO: Jaggedize.
        pixel_thresholds(: obj: `numpy.ndarray`): list of discriminator
            thresholds for each pixel. Shape (n_unique_pix,)
    """
    ip = cuda.grid(1)

    # equivalent to num_backtrack.sum()
    total_backtracks = offset_backtrack[-1] + num_backtrack[-1]

    ntrks = min(num_backtrack[ip], current_fractions.shape[2])

    if ip < pixels_signals.shape[0]:
        curre = pixels_signals[ip]
        ic = 0
        iadc = 0
        adc_busy = 0
        last_reset = 0
        true_q = 0
        q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e

        while ic < curre.shape[0] or adc_busy > 0:

            if iadc >= sim.MAX_ADC_VALUES:
                print("More ADC values than possible,", sim.MAX_ADC_VALUES)
                break

            q = 0
            if detector.BUFFER_RISETIME > 0:
                conv_start = max(last_reset, floor(ic - 10*detector.BUFFER_RISETIME/detector.TIME_SAMPLING))
                for jc in range(conv_start, min(ic+1, curre.shape[0])):
                    w = exp((jc - ic) * detector.TIME_SAMPLING / detector.BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/detector.BUFFER_RISETIME))
                    q += curre[jc] * detector.TIME_SAMPLING * w

                    for itrk in range(ntrks):
                        idx = total_backtracks * jc + offset_backtrack[ip] + itrk
                        current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING * w

            elif ic < curre.shape[0]:
                q += curre[ic] * detector.TIME_SAMPLING
                for itrk in range(ntrks):
                    idx = total_backtracks * ic + offset_backtrack[ip] + itrk
                    current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING

            q_sum += q
            true_q += q

            q_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.UNCORRELATED_NOISE_CHARGE * e
            disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.DISCRIMINATOR_NOISE * e

            if adc_busy > 0:
                adc_busy -= 1

            if q_sum + q_noise >= pixel_thresholds[ip] + disc_noise and adc_busy == 0:
                interval = round((3 * detector.CLOCK_CYCLE + detector.ADC_HOLD_DELAY * detector.CLOCK_CYCLE) / detector.TIME_SAMPLING)
                integrate_end = ic+interval

                ic+=1

                while ic <= integrate_end:
                    q = 0

                    if detector.BUFFER_RISETIME > 0:
                        conv_start = max(last_reset, floor(ic - 10*detector.BUFFER_RISETIME/detector.TIME_SAMPLING))
                        for jc in range(conv_start, min(ic+1, curre.shape[0])):
                            w = exp((jc - ic) * detector.TIME_SAMPLING / detector.BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/detector.BUFFER_RISETIME))
                            q += curre[jc] * detector.TIME_SAMPLING * w

                            for itrk in range(ntrks):
                                idx = total_backtracks * jc + offset_backtrack[ip] + itrk
                                current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING * w

                    elif ic < curre.shape[0]:
                        q += curre[ic] * detector.TIME_SAMPLING
                        for itrk in range(ntrks):
                            idx = total_backtracks * ic + offset_backtrack[ip] + itrk
                            current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING

                    q_sum += q
                    true_q += q
                    ic+=1

                adc = q_sum + xoroshiro128p_normal_float32(rng_states, ip) * detector.UNCORRELATED_NOISE_CHARGE * e
                disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.DISCRIMINATOR_NOISE * e

                if adc < pixel_thresholds[ip] + disc_noise:
                    ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
                    q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e
                    true_q = 0

                    for itrk in range(current_fractions.shape[2]):
                        current_fractions[ip][iadc][itrk] = 0
                    last_reset = ic
                    continue

                #tot_backtracked = 0
                #for itrk in range(current_fractions.shape[2]):
                #    tot_backtracked += current_fractions[ip][iadc][itrk]

                if true_q > 0:
                    for itrk in range(current_fractions.shape[2]):
                        current_fractions[ip][iadc][itrk] /= true_q

                adc_list[ip][iadc] = adc

                crossing_time_tick = min((ic, len(time_ticks)-1))
                # handle case when tick extends past end of current array
                post_adc_ticks = max((ic - crossing_time_tick, 0))
                #+2-tick delay from when the PACMAN receives the trigger and when it registers it.
                adc_ticks_list[ip][iadc] = time_ticks[crossing_time_tick]+time_padding-2+post_adc_ticks

                ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
                last_reset = ic
                adc_busy = round(detector.ADC_BUSY_DELAY * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)

                q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e
                true_q = 0

                iadc += 1
                continue

            ic += 1
