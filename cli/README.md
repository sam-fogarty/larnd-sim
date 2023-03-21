To simulate a radiological (e.g. 39Ar), we must create a geant4 macro for edep-sim. 

`tgraph_to_hist.py` takes as input a ROOT file containing TGraphs of energy spectra (energy in keV vs y-val). To produces a macro that creates an arbitrary energy histogram that represents the spectra. In command-line: 

`python3 tgraph_to_hist.py <path_to_ROOT_file_folder> <ROOT_filename> <path_to_macro_folder>`

Run `edep-sim` in command-line:

`edep-sim -e <num_of_events> -p QGSP_BERT_LIV -o <path_to_output_file> <path_to_macro>`

Run `dumpTree.py` to convert ROOT to h5:

`python3 dumpTree.py <path_to_ROOT_file> <path_to_h5_file>`

Run `simulate_pixels.py` to simulate detector response:

`python3 simulate_pixels.py \
--input_filename= <path_to_edep_h5_file> \
--output_filename= <path_to_larndsim_output_h5_file> \
--detector_properties=../larndsim_NEST/detector_properties/module0.yaml \
--simulation_properties=../larndsim_NEST/simulation_properties/singles_sim.yaml \
--pixel_layout=../larndsim_NEST/pixel_layouts/multi_tile_layout-2.3.16.yaml \
--response_file=../larndsim_NEST/bin/response_44.npy \
--light_lut_filename=../larndsim_NEST/bin/lightLUT.npz \
--light_det_noise_filename=../larndsim_NEST/bin/light_noise-module0.npy \
--pixel_thresholds_file=<path_to_pixel_thresholds_file>`

`pixel_thresholds_file` is optional, but provides a way to match the thresholds channel-by-channel between data and MC. Note that you will need to change `detector_properties`, `pixel_layout`, `response_file`, `light_det_noise_filename`, and `pixel_thresholds_file` depending on the detector being simulated.


