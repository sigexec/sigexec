"""Diagnostics and visualization utilities for sigchain."""

from .visualization import (
    plot_timeseries,
    plot_pulse_matrix,
    plot_range_profile,
    plot_range_doppler_map,
    plot_spectrum,
    create_comparison_plot,
)

from .plot_blocks import (
    add_timeseries_plot,
    add_pulse_matrix_plot,
    add_range_profile_plot,
    add_range_doppler_map_plot,
    add_spectrum_plot,
)

__all__ = [
    "plot_timeseries",
    "plot_pulse_matrix",
    "plot_range_profile",
    "plot_range_doppler_map",
    "plot_spectrum",
    "create_comparison_plot",
    "add_timeseries_plot",
    "add_pulse_matrix_plot",
    "add_range_profile_plot",
    "add_range_doppler_map_plot",
    "add_spectrum_plot",
]
