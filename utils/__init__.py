# utils/__init__.py
from utils.c3d_loader import (
    load_c3d, get_file_metadata,
    get_marker_data, get_analog_data, get_force_plate_data,
)
from utils.marker_analysis import (
    butterworth_filter, filter_marker_df,
    compute_velocity, compute_acceleration, compute_speed,
    compute_distance_between, compute_angle_3pt_safe,
    marker_stats, detect_gaps,
)
from utils.analog_analysis import (
    bandpass_filter, lowpass_filter, highpass_filter, notch_filter,
    process_emg, emg_stats, compute_psd, compute_fft,
    compute_impulse, compute_loading_rate,
    detect_stance_phases, cop_path_length,
    detect_gait_events, compute_gait_metrics,
)
from utils.visualization import (
    PALETTE,
    plot_marker_trajectories, plot_marker_speed,
    plot_3d_trajectories, plot_stick_figure, plot_marker_heatmap,
    plot_marker_range_bar,
    plot_analog_channels, plot_grf, plot_cop,
    plot_emg, plot_psd,
    plot_stance_phases, plot_correlation_matrix,
)
