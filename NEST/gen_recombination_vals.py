import numpy as np
import fire
import nestpy
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(start_keV, end_keV, step_size, num_events, efield):
    # calculate average recombination values for an energy range, Efield, and number of events
    energy_vals = np.arange(start_keV,end_keV+step_size, step_size)
    detector = nestpy.VDetector()
    larnest = nestpy.LArNEST(detector)
    density = 1.393
    print('Detector set to LAr detector.')
    Nph_all = []
    Ne_all = []
    for i in tqdm(range(len(energy_vals)), desc=' Calculating R for each energy value: '):
        energy_step = energy_vals[i]
        Nph = np.zeros(num_events)
        Ne = np.zeros(num_events)
        
        # calculate number of photons and number of electrons for specific energy, with fluctuations
        for ii in range(num_events):
            result = larnest.full_calculation(
                nestpy.LArInteraction.ER,
                energy_step,
                efield,
                density,
                False
            )
            Nph[ii] = result.fluctuations.NphFluctuation
            Ne[ii] = result.fluctuations.NeFluctuation
        
        # calculate mean values of all events 
        Nph_mean = np.mean(Nph)
        Ne_mean = np.mean(Ne)

        Nph_std = np.std(Nph) / np.sqrt(num_events)
        Ne_std = np.std(Ne) / np.sqrt(num_events)

        Nph_all.append(Nph_mean)
        Ne_all.append(Ne_mean)
    
    # save recombination values to a text file for larnd-sim 
    recomb = np.array(Ne_all)/(np.array(Ne_all) + np.array(Nph_all))
    filepath_txt = 'NEST_R-values_' + 'efield' + str(efield) + '_' + \
        str(start_keV) + 'keV_to_' + str(end_keV) + 'keV_' + str(step_size) + 'keV-stepsize_' +\
        str(num_events) + '-events' + '.txt'
    print('Writing text file ', filepath_txt)
    f = open(filepath_txt,"w+")
    for i, R in enumerate(recomb):
        E = energy_vals[i]
        f.write(str(E) + ',' + str(R))
        f.write('\n')
    f.close()
    
    # load in text file
    data = np.loadtxt(filepath_txt, dtype='float', delimiter=',')
    energies = data[:,0]
    recombination = data[:,1]
    
    # make plot and save pdf
    plt.plot(energies, recombination)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Recombination Factor R')
    plt.savefig(filepath_txt.split('.')[0] + '.pdf')
    
    filepath_h5 = filepath_txt.split('.')[0] + '.h5'
    print('Writing h5 file ', filepath_h5)
    # save recombination values and energies in h5 file
    dtype = np.dtype([('E_start', 'f4'),('R', 'f4')])
    datapoints = np.empty(len(energies), dtype=dtype)
    for i in range(len(energies)):
        datapoints[i]['E_start'] = energies[i]
        datapoints[i]['R'] = recombination[i]

    with h5py.File(filepath_h5, 'w') as f:
        f.create_dataset("NEST", data=datapoints)
    

if __name__ == "__main__":
    fire.Fire(main)