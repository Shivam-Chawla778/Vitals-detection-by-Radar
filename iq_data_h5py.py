import numpy as np
import h5py

num_frames=1316
sweeps_per_frame=16
num_points=21

with h5py.File("r3.h5", "r") as f:
   # Access the dataset
   raw_iq = f['sessions']['session_0']['group_0']['entry_0']['result']['frame'][:]
   # Convert to complex numbers 
   iq_data = raw_iq['real'].astype(np.float32) + 1j * raw_iq['imag'].astype(np.float32)

iq_data_ds=np.zeros((num_frames,sweeps_per_frame,num_points),dtype=complex)
# Downsampling of data
for i in range(num_frames):
    iq_data_ds[i]=iq_data[i]
print(iq_data_ds)
