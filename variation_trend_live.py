# this code applies variation trend method to detect the phase signal vs time 
# plot , so that the plot can be used further for analysis 


import numpy as np
import matplotlib.pyplot as plt
from acconeer.exptool import a121
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Loading data
def main(duration):
    
     # Connect to radar over UART (USB)
     client = a121.Client.open(serial_port="/dev/ttyUSB0")
     # Config params
     fs = 36  # Sweep rate (Hz)
     range_spacing = 0.5e-3  # Range spacing (m)
     D = 100                # Downsampling factor
     tau_iq = 0.04          # Time constant for low-pass filter (seconds)
     f_low = 0.2            # High-pass filter cutoff frequency (Hz)
     sweeps_per_frame = 32  # Number of sweeps per frame
     num_points=40
     num_frames = fs * duration
       
 
     # Sensor config
     config = a121.SensorConfig(
         profile=a121.Profile.PROFILE_3,
         frame_rate=fs,
         sweeps_per_frame=sweeps_per_frame,
         num_points=num_points
         )
 
 
     # Setup and start
     client.setup_session(config)
     client.start_session()
     
    # Initialize
    # iq_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
    # iq_avg_data = np.zeros((num_frames, num_points), dtype=complex)
     IQ_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
     print("Collecting data...")
     init_time = time.time()
     
     for i in range(num_frames):
        result = client.get_next()
        IQ_data[i]=result.frame
        
     end_time = time.time()
     print("Done recording")
     client.stop_session()
     client.close()
     time_taken = end_time - init_time
     print(f"Data collection took {time_taken:.2f} seconds")
     
     # Transpose data to match MATLAB's order: (frames x range bins x sweeps)
     IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)
     
     # Compute the magnitude of IQ data (sweeps x range bins)
     magnitude_data = np.abs(IQ_data)
     
     # Finding the range bin with the highest peak magnitude (across all sweeps)
     mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
     peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna
     
     # Select the range indices based on the peak range bin
     range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
     range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
     range_indices = np.arange(range_start_bin, range_end_bin + 1)
     
     # Downsampling
     downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)
     
     # Temporal low-pass filter parameters
     alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient
     
     # Initializing filtered data
     filtered_data = np.zeros_like(downsampled_data)
     filtered_data[:, :, 0] = downsampled_data[:, :, 0]
     
     # Apply temporal low-pass filter
     for s in range(1, downsampled_data.shape[2]):
         filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                                  (1 - alpha_iq) * downsampled_data[:, :, s]
     
     # Phase unwrapping and high-pass filtering parameters
     alpha_phi = np.exp(-2 * f_low / fs)  # High-pass filter coefficient
     
     # Initialize phase values
     phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep
     
     # Calculate phase for each sweep
     for s in range(1, filtered_data.shape[2]):
         z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
         phi[s] = alpha_phi * phi[s - 1] + np.angle(z)
    
     
     # Plot the phase vs. frames with expanded x-axis
     plt.figure(figsize=(12, 6))  # Increase figure width for better spacing
     plt.plot(range(len(phi)), phi, linewidth=1.5 , color ="black")
     plt.xticks(np.arange(0, len(phi), step=100))  # Set x-axis ticks at intervals of 100
     plt.xlabel('Frame Index (sweeps)')
     plt.ylabel('Phase (radians)')
     plt.title('Variation Trend Method : Demodualted Signal')
     # plt.grid(True)
     plt.tight_layout()  # Ensure no clipping
     plt.show()

     peaks, properties = find_peaks(phi, 
                                  prominence=np.std(phi)*0.3,
                                  distance=(1))  # Min 0.25s between peaks
     
     # Calculate BPM
     if len(peaks) > 1:
         
         bpm = len(peaks) * (60 / time_taken)  # peaks per minute
         print(f"Detected BPM: {bpm:.1f}")
         print(f"Number of peaks found: {len(peaks)}")
     else:
         print("Insufficient peaks detected")

if __name__=="__main__":
    main(duration=10)      
     
