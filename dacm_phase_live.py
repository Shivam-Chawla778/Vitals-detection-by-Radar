#import numpy as np
#import matplotlib.pyplot as plt
#from acconeer.exptool import a121
#from scipy.signal import  butter, filtfilt
#from attr import evolve
#def bandpass_filter(signal, low, high, fs, order=3):
#    nyq = fs / 2
#    b, a = butter(order, [low / nyq, high / nyq], btype='band')
#    return filtfilt(b, a, signal)
#
#def main():
#    # Connect to radar over UART (USB)
#    client = a121.Client.open(serial_port="/dev/ttyUSB0")
#
#    # Config params
#    frame_rate = 20  # Hz
#    duration = 20
#    # seconds
#    num_frames = frame_rate * duration
#    sweeps_per_frame = 32
#    num_points = 100
#
#    # Sensor config
#    config = a121.SensorConfig(
#        profile=a121.Profile.PROFILE_3,
#        frame_rate=frame_rate,
#        start_point=50,
#        num_points=num_points,
#        step_length=1,
#        sweeps_per_frame=sweeps_per_frame,
#    )
#
#    # Setup and start
#    client.setup_session(config)
#    client.start_session()
#    
#    
#    # Initialize
#    iq_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
#    iq_avg_data = np.zeros((num_frames, num_points), dtype=complex)
#    max_bin_idx_series = []
#    
#    # Find the chest bin using first few frames for stability
#    initial_frames = min(50, num_frames)
#    amplitude_avg = np.mean(np.abs(iq_avg_data[:initial_frames]), axis=0)
#    chest_bin = np.argmax(amplitude_avg)
#    
#    print("collecting data")
#    for i in range(num_frames):
#        result = client.get_next()
#        iq_data[i] = result.frame  # shape: (sweeps_per_frame, num_points)
#        iq_avg_data[i]=np.mean(iq_data[i], axis=0)
#        iq_avg_data[i]=iq_avg_data[i][np.newaxis,:]  #shape: (1,num_points)
#        #result = evolve(result, frame=result.frame  ) # function to make an updated copy of an immutable object
#                                                      
#                  #shape : (frames,avged_sweep(1),num_points)
#        # Get amplitude and find chest bin
#       # shape: (1,num_points)
#    
#    # Find the chest bin using first few frames for stability
#    initial_frames = min(50, num_frames)
#    amplitude_avg = np.mean(np.abs(iq_avg_data[:initial_frames]), axis=0)
#    chest_bin = np.argmax(amplitude_avg) 
#    print("done recording")
#    client.stop_session()
#    client.close()
#
#    # Convert max_bin_idx_series list to numpy array
#    max_bin_idx_series=np.array(max_bin_idx_series)
#    
#    phase_series=np.zeros(num_frames)
#    i_avg_data=np.array(np.real(iq_avg_data))
#    q_avg_data=np.array(np.imag(iq_avg_data))
#    
#    
#   
#    for j in range(0,num_frames-1):
#     if (j==0):
#        phase_series[0]=0.1
#        continue
#     else:
#          frame_idx=j
#          chase=((  ( i_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]] ) * (( q_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]]) - ( q_avg_data[frame_idx-1][0][max_bin_idx_series[frame_idx-1]] ))  )        -         (  ( q_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]] ) *  (( i_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]]) - ( i_avg_data[frame_idx-1][0][max_bin_idx_series[frame_idx-1]] )) ) )      /        ( (( i_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]] )**2) +  ( (q_avg_data[frame_idx][0][max_bin_idx_series[frame_idx]])**2) + 1e-9 ) 
#          phase_series[frame_idx]=phase_series[frame_idx-1] + chase
#                 
#     
#    # Filter to remove noise and isolate breathing frequencies (e.g., 0.1–0.5 Hz)
#    filtered_phase = bandpass_filter(phase_series, 0.1, 0.5, fs=frame_rate)
#    # Plot
#    time_axis = np.linspace(0, duration, num_frames)
#    plt.plot(time_axis,filtered_phase)
#    plt.xlabel("Time (s)")
#    plt.ylabel("Phase (radians)")
#    plt.title("filtered Phase demodulation by dacm algorithm tracking max amplitude range bin")
#    plt.grid(True)
#    plt.show()
#if  __name__== "__main__":
#    main()
import numpy as np
import matplotlib.pyplot as plt
from acconeer.exptool import a121
from scipy.signal import butter, filtfilt, find_peaks
from attr import evolve

def bandpass_filter(signal, low, high, fs, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def main():
    # Connect to radar over UART (USB)
    client = a121.Client.open(serial_port="/dev/ttyUSB0")

    # Config params
    frame_rate = 20  # Hz
    duration = 30  # seconds
    num_frames = frame_rate * duration
    sweeps_per_frame = 32
    num_points = 100

    # Sensor config
    config = a121.SensorConfig(
        profile=a121.Profile.PROFILE_3,
        frame_rate=frame_rate,
        start_point=50,
        num_points=num_points,
        step_length=1,
        sweeps_per_frame=sweeps_per_frame,
    )

    # Setup and start
    client.setup_session(config)
    client.start_session()
    
    # Initialize
    iq_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
    iq_avg_data = np.zeros((num_frames, num_points), dtype=complex)
    
    print("Collecting data...")
    for i in range(num_frames):
        result = client.get_next()
        iq_data[i] = result.frame
        # Average across sweeps - simplified indexing
        iq_avg_data[i] = np.mean(iq_data[i], axis=0)
    
    print("Done recording")
    client.stop_session()
    client.close()

    # Find the chest bin using first few frames for stability
    initial_frames = min(50, num_frames)
    amplitude_avg = np.mean(np.abs(iq_avg_data[:initial_frames]), axis=0)
    chest_bin = np.argmax(amplitude_avg)
    print(f"Using chest bin: {chest_bin}")
    
    # Extract I and Q for the fixed chest bin
    i_data = np.real(iq_avg_data[:, chest_bin])
    q_data = np.imag(iq_avg_data[:, chest_bin])
    
    # Calculate phase using DACM algorithm - FIXED VERSION
    phase_series = np.zeros(num_frames)
    
    for j in range(1, num_frames):
        # Proper DACM phase difference calculation
        numerator = (i_data[j] * (q_data[j] - q_data[j-1]) - 
                    q_data[j] * (i_data[j] - i_data[j-1]))
        denominator = (i_data[j]**2 + q_data[j]**2 + 1e-12)
        
        phase_diff = numerator / denominator
        phase_series[j] = phase_series[j-1] + phase_diff
    
    # FIXED: Use broader filter for higher BPM detection
    # 0.2-2.0 Hz covers 12-120 BPM range
    filtered_phase = bandpass_filter(phase_series, 0.2, 2.0, fs=frame_rate)
    
    # Additional detrending to remove low-frequency drift
    from scipy.signal import detrend
    filtered_phase = detrend(filtered_phase)
    
    # Peak detection for BPM calculation
    # Adjust prominence and distance based on expected breathing rate
    peaks, properties = find_peaks(filtered_phase, 
                                 prominence=np.std(filtered_phase)*0.3,
                                 distance=frame_rate//4)  # Min 0.25s between peaks
    
    # Calculate BPM
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / frame_rate  # intervals in seconds
        avg_interval = np.mean(peak_intervals)
        bpm = 60 / avg_interval
        print(f"Detected BPM: {bpm:.1f}")
        print(f"Number of peaks found: {len(peaks)}")
    else:
        print("Insufficient peaks detected")
    
    # Plot results
    time_axis = np.linspace(0, duration, num_frames)
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Raw phase
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, phase_series)
    plt.title('Raw Phase Signal')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    
    # Plot 2: Filtered phase with peaks
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, filtered_phase)
    plt.plot(time_axis[peaks], filtered_phase[peaks], 'ro', markersize=8)
    plt.title(f'Filtered Phase Signal (Peaks: {len(peaks)})')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.show()
    ## Plot 3: Amplitude at chest bin
    #plt.subplot(3, 1, 3)
    #amplitude_series = np.abs(iq_avg_data[:, chest_bin])
    #plt.plot(time_axis, amplitude_series)
    #plt.title(f'Amplitude at Chest Bin {chest_bin}')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Amplitude')
    #plt.grid(True)
    #
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()