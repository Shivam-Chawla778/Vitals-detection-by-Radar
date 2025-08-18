import numpy as np
import matplotlib.pyplot as plt
import time
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
    distance=float(input("Enter the distance in meters: "))
    # Config params
    frame_rate = 20
    buffer= 0.5  # Hz
    duration = 30  # seconds
    num_frames = frame_rate * duration
    sweeps_per_frame = 8
    num_points = int((2*buffer)/0.0025)  
    start_point = int((distance-buffer)/0.0025)  # Convert distance to start point index

    # Sensor config
    config = a121.SensorConfig(
        profile=a121.Profile.PROFILE_3,
        frame_rate=frame_rate,
        start_point=start_point,
        step_length=1,
        num_points=num_points,
        sweeps_per_frame=sweeps_per_frame,
    )


    # Setup and start
    client.setup_session(config)
    client.start_session()
    
    # Initialize
    iq_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
    iq_avg_data = np.zeros((num_frames, num_points), dtype=complex)
    
    print("Collecting data...")
    init_time = time.time()
    for i in range(num_frames):
        result = client.get_next()
        iq_data[i] = result.frame
        # Average across sweeps - simplified indexing
        iq_avg_data[i] = np.mean(iq_data[i], axis=0)
    
    end_time = time.time()
    print("Done recording")
    client.stop_session()
    client.close()
    time_taken = end_time - init_time
    print(f"Data collection took {time_taken:.2f} seconds")
    # Find the chest bin using first few frames for stability
    initial_frames = min(50, num_frames)
    amplitude_avg = np.mean(np.abs(iq_avg_data[:initial_frames]), axis=0)
    chest_bin = np.argmax(amplitude_avg)
    print(f"Using chest bin: {chest_bin}")
    
    # Extract I and Q for the fixed chest bin
    i_data = np.real(iq_avg_data[:, chest_bin])
    q_data = np.imag(iq_avg_data[:, chest_bin])
    
    
    phase_series = np.zeros(num_frames)
    
    for j in range(1, num_frames):
        
        numerator = (i_data[j] * (q_data[j] - q_data[j-1]) - 
                    q_data[j] * (i_data[j] - i_data[j-1]))
        denominator = (i_data[j]**2 + q_data[j]**2 + 1e-12)
        
        phase_diff = numerator / denominator
        phase_series[j] = phase_series[j-1] + phase_diff
    
    
    # 0.2-2.0 Hz covers 12-120 BPM range
    filtered_phase = bandpass_filter(phase_series, 0.2, 2.0, fs=frame_rate)
    
    # Additional detrending to remove low-frequency drift
    from scipy.signal import detrend
    filtered_phase = detrend(filtered_phase)
    
    # Peak detection for BPM calculation
    # Adjust prominence and distance based on expected breathing rate
    peaks, properties = find_peaks(filtered_phase, 
                                 prominence=np.std(filtered_phase)*0.3,
                                 distance=1)
                                 #distance=frame_rate//4)  # Min 0.25s between peaks
    
    # Calculate BPM
    if len(peaks) > 1:
        #peak_intervals = np.diff(peaks) / frame_rate  # intervals in seconds
        #avg_interval = np.mean(peak_intervals)
        #bpm = 60 / avg_interval
        bpm = len(peaks) * (60 / time_taken)  # peaks per minute
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
    # Plot 3: Amplitude at chest bin
    plt.subplot(3, 1, 3)
    amplitude_series = np.abs(iq_avg_data[:, chest_bin])
    plt.plot(time_axis, amplitude_series)
    plt.title(f'Amplitude at Chest Bin {chest_bin}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
