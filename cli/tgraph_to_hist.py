import ROOT
import matplotlib.pyplot as plt
import numpy as np
import fire
#import os

def main(folder_for_ROOT_files, ROOT_filename, folder_for_macros, event_type, binsize_rebin=1):
    ### script to convert TGraph in ROOT file to a geant4 macro to use in edep-sim for radiologicals
    ROOT_filepath = folder_for_ROOT_files + '/' + ROOT_filename
    print('Opening '+ ROOT_filepath +'...')
    root_file = ROOT.TFile.Open(ROOT_filepath,"READ")
    
    # Get the TGraph object from the file
    if event_type == 'Betas':
        graph = root_file.Get("Betas")
    elif event_type == 'Gammas':
        graph = root_file.Get("Gammas")
    else:
        raise ValueError('Possible values for event_type are Betas and Gammas')
    nPoints = graph.GetN()
    print('Number of points in TGraph = ', nPoints)
    print('Rebinning to ', binsize_rebin, ' keV...')
    # get x and y values in graph
    x = np.array(graph.GetX()) # keV
    y = np.array(graph.GetY())
    
    # rebin
    x_rebin = np.arange(np.min(x), np.max(x), binsize_rebin)
    y_norm_rebin = np.interp(x_rebin, x, y/np.sum(y))
    
    # make macro for edep-sim
    if event_type == 'Betas':
        filepath_txt = folder_for_macros + '/' + ROOT_filename.split('.')[0] + '_betas' + '.mac'
    if event_type == 'Gammas':
        filepath_txt = folder_for_macros + '/' + ROOT_filename.split('.')[0] + '_gammas' + '.mac'
    print('Creating macro at ', filepath_txt+' ...')
    #if os.path.exists(filepath_txt): 
    #    raise Exception('Output file '+ str(filepath_txt) + ' already exists.')
    f = open(filepath_txt, "w+")
    f.write('/edep/random/timeRandomSeed\n')
    f.write('/edep/gdml/read /sdf/home/s/sfogarty/Desktop/LArTPC_sim/geometries/Module0.gdml\n')
    f.write('/edep/hitLength TPCActive_shape 0.00001 mm\n')
    f.write('/process/eLoss/StepFunction 0.2 0.1  mm\n')
    f.write('/edep/update\n')
    f.write('/gps/pos/type Volume\n')
    f.write('/gps/pos/shape Para\n')
    f.write('/gps/pos/centre 0.0 -22 0.0 cm\n')
    f.write('/gps/pos/halfx 30.2723 cm\n')
    f.write('/gps/pos/halfy 62.0543 cm\n')
    f.write('/gps/pos/halfz 31.0163 cm\n')
    f.write('/gps/ang/type iso\n')
    if event_type == 'Betas':
        f.write('/gps/particle e-\n')
    elif event_type == 'Gammas':
        f.write('/gps/particle gamma\n')
    f.write('/gps/ene/type Arb\n')
    f.write('/gps/hist/type arb\n')

    for i in range(len(x_rebin)):
        f.write('/gps/hist/point '+str("%.6f" % (x_rebin[i]*1e-3))+' '+str("%.6f" % y_norm_rebin[i]) + '\n')

    f.write('/gps/hist/inter Lin')
    f.close()
    root_file.Close()
if __name__ == "__main__":
    fire.Fire(main)