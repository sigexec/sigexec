"""
Visualization utilities for signal processing using Plotly.

This module provides functions to create interactive Plotly plots
that can be used standalone or with staticdash for dashboard creation.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Tuple
from sigchain.core.data import SignalData


def plot_timeseries(
    signal_data: SignalData,
    title: str = "Signal Timeseries",
    show_real: bool = True,
    show_imag: bool = True,
    show_magnitude: bool = False,
    max_samples: int = 10000,
    height: int = 400,
) -> go.Figure:
    """
    Create an interactive timeseries plot of signal data.
    
    Args:
        signal_data: SignalData object to plot
        title: Plot title
        show_real: Show real component
        show_imag: Show imaginary component  
        show_magnitude: Show magnitude
        max_samples: Maximum number of samples to plot (for performance)
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    data = signal_data.data
    
    # Flatten if multidimensional (take first row for display)
    if data.ndim > 1:
        data = data[0, :]
    
    # Downsample if too many points
    if len(data) > max_samples:
        indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
        data = data[indices]
    else:
        indices = np.arange(len(data))
    
    # Create time axis
    time = indices / signal_data.sample_rate * 1e6  # Convert to microseconds
    
    fig = go.Figure()
    
    if show_real:
        fig.add_trace(go.Scatter(
            x=time,
            y=np.real(data),
            mode='lines',
            name='Real',
            line=dict(color='blue', width=1),
        ))
    
    if show_imag and np.iscomplexobj(data):
        fig.add_trace(go.Scatter(
            x=time,
            y=np.imag(data),
            mode='lines',
            name='Imaginary',
            line=dict(color='red', width=1),
        ))
    
    if show_magnitude:
        fig.add_trace(go.Scatter(
            x=time,
            y=np.abs(data),
            mode='lines',
            name='Magnitude',
            line=dict(color='green', width=2),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (μs)",
        yaxis_title="Amplitude",
        hovermode='x unified',
        height=height,
        template='plotly_white',
    )
    
    return fig


def plot_pulse_matrix(
    signal_data: SignalData,
    title: str = "Pulse Matrix",
    colorscale: str = "Greys",
    height: int = 500,
    use_db: bool = True,
    db_floor: float = -60,
) -> go.Figure:
    """
    Create a heatmap visualization of pulse data (2D matrix).
    
    Args:
        signal_data: SignalData object with 2D data (pulses x samples)
        title: Plot title
        colorscale: Plotly colorscale name
        height: Plot height in pixels
        use_db: Convert to dB scale
        db_floor: Floor value for dB conversion
        
    Returns:
        Plotly Figure object
    """
    data = signal_data.data
    
    if data.ndim != 2:
        raise ValueError("Data must be 2D for pulse matrix plot")
    
    # Convert to magnitude
    magnitude = np.abs(data)
    
    if use_db:
        z_data = 20 * np.log10(magnitude + 1e-10)
        z_data = np.maximum(z_data, db_floor)
        colorbar_title = "Magnitude (dB)"
    else:
        z_data = magnitude
        colorbar_title = "Magnitude"
    
    # Create sample and pulse axes
    num_pulses, num_samples = data.shape
    sample_indices = np.arange(num_samples)
    pulse_indices = np.arange(num_pulses)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=sample_indices,
        y=pulse_indices,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Pulse Index",
        height=height,
        template='plotly_white',
    )
    
    return fig


def plot_range_profile(
    signal_data: SignalData,
    title: str = "Range Profile",
    pulse_index: Optional[int] = None,
    use_db: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Plot range profile (single pulse or averaged).
    
    Args:
        signal_data: SignalData with range-compressed data
        title: Plot title
        pulse_index: Specific pulse to plot (None = average all)
        use_db: Convert to dB scale
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    data = signal_data.data
    
    if data.ndim == 1:
        profile = data
    elif pulse_index is not None:
        profile = data[pulse_index, :]
    else:
        # Average over all pulses
        profile = np.mean(np.abs(data), axis=0)
    
    magnitude = np.abs(profile)
    
    if use_db:
        y_data = 20 * np.log10(magnitude + 1e-10)
        y_label = "Magnitude (dB)"
    else:
        y_data = magnitude
        y_label = "Magnitude"
    
    # Create range axis if metadata available
    if 'samples_per_pulse' in signal_data.metadata:
        # Convert sample index to range (km)
        c = 3e8  # Speed of light
        range_axis = np.arange(len(y_data)) / signal_data.sample_rate * c / 2 / 1000
        x_label = "Range (km)"
        x_data = range_axis
    else:
        x_label = "Sample Index"
        x_data = np.arange(len(y_data))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Profile',
    ))
    
    # Mark target if metadata available
    if 'target_delay' in signal_data.metadata and 'samples_per_pulse' in signal_data.metadata:
        target_range = signal_data.metadata['target_delay'] * c / 2 / 1000
        fig.add_vline(
            x=target_range,
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        template='plotly_white',
        hovermode='x',
    )
    
    return fig


def plot_range_doppler_map(
    signal_data: SignalData,
    title: str = "Range-Doppler Map",
    colorscale: str = "Greys",
    height: int = 600,
    use_db: bool = True,
    db_range: float = 50,
    mark_target: bool = True,
) -> go.Figure:
    """
    Create an interactive range-Doppler map visualization.
    
    Args:
        signal_data: SignalData with range-doppler map
        title: Plot title
        colorscale: Plotly colorscale name
        height: Plot height in pixels
        use_db: Convert to dB scale
        db_range: Dynamic range in dB
        mark_target: Mark true target location if available
        
    Returns:
        Plotly Figure object
    """
    data = signal_data.data
    
    if data.ndim != 2:
        raise ValueError("Data must be 2D for range-Doppler map")
    
    # Convert to dB
    magnitude = np.abs(data)
    
    if use_db:
        z_data = 20 * np.log10(magnitude + 1e-10)
        max_val = np.max(z_data)
        z_data = np.maximum(z_data, max_val - db_range)
        colorbar_title = "Magnitude (dB)"
    else:
        z_data = magnitude
        colorbar_title = "Magnitude"
    
    # Create axes
    c = 3e8  # Speed of light
    num_doppler, num_range = data.shape
    
    # Range axis (km)
    range_axis = np.arange(num_range) / signal_data.sample_rate * c / 2 / 1000
    
    # Doppler axis (Hz)
    if 'doppler_frequencies' in signal_data.metadata:
        doppler_axis = signal_data.metadata['doppler_frequencies']
    else:
        doppler_axis = np.arange(num_doppler) - num_doppler // 2
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=range_axis,
        y=doppler_axis,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
    ))
    
    # Mark target if available
    if mark_target and 'target_delay' in signal_data.metadata and 'target_doppler' in signal_data.metadata:
        target_range = signal_data.metadata['target_delay'] * c / 2 / 1000
        target_doppler = signal_data.metadata['target_doppler']
        
        fig.add_trace(go.Scatter(
            x=[target_range],
            y=[target_doppler],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color='rgba(0,0,0,0)',  # Transparent fill
                line=dict(width=2, color='black'),
            ),
            name='True Target',
            showlegend=True,
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Range (km)",
        yaxis_title="Doppler Frequency (Hz)",
        height=height,
        template='plotly_white',
    )
    
    return fig


def plot_spectrum(
    signal_data: SignalData,
    title: str = "Frequency Spectrum",
    use_db: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Plot frequency spectrum of signal.
    
    Args:
        signal_data: SignalData object
        title: Plot title
        use_db: Convert to dB scale
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    data = signal_data.data
    
    # Flatten if multidimensional
    if data.ndim > 1:
        data = data.flatten()
    
    # Compute FFT
    spectrum = np.fft.fftshift(np.fft.fft(data))
    magnitude = np.abs(spectrum)
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1/signal_data.sample_rate))
    freqs = freqs / 1e6  # Convert to MHz
    
    if use_db:
        y_data = 20 * np.log10(magnitude + 1e-10)
        y_label = "Magnitude (dB)"
    else:
        y_data = magnitude
        y_label = "Magnitude"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=y_data,
        mode='lines',
        line=dict(color='purple', width=1),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (MHz)",
        yaxis_title=y_label,
        height=height,
        template='plotly_white',
        hovermode='x',
    )
    
    return fig


def create_comparison_plot(
    signals: list[SignalData],
    labels: list[str],
    title: str = "Signal Comparison",
    plot_type: str = "timeseries",
    height: int = 500,
) -> go.Figure:
    """
    Create a comparison plot of multiple signals.
    
    Args:
        signals: List of SignalData objects
        labels: List of labels for each signal
        title: Plot title
        plot_type: Type of plot ('timeseries', 'spectrum', 'magnitude')
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (signal, label) in enumerate(zip(signals, labels)):
        data = signal.data
        
        # Flatten if needed
        if data.ndim > 1:
            data = data[0, :]
        
        if plot_type == "timeseries":
            time = np.arange(len(data)) / signal.sample_rate * 1e6
            fig.add_trace(go.Scatter(
                x=time,
                y=np.real(data),
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)]),
            ))
            x_label = "Time (μs)"
            y_label = "Amplitude"
            
        elif plot_type == "spectrum":
            spectrum = np.fft.fftshift(np.fft.fft(data))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1/signal.sample_rate)) / 1e6
            fig.add_trace(go.Scatter(
                x=freqs,
                y=20 * np.log10(np.abs(spectrum) + 1e-10),
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)]),
            ))
            x_label = "Frequency (MHz)"
            y_label = "Magnitude (dB)"
            
        elif plot_type == "magnitude":
            time = np.arange(len(data)) / signal.sample_rate * 1e6
            fig.add_trace(go.Scatter(
                x=time,
                y=np.abs(data),
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)]),
            ))
            x_label = "Time (μs)"
            y_label = "Magnitude"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        template='plotly_white',
        hovermode='x unified',
    )
    
    return fig
