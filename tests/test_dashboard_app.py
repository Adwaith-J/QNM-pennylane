import numpy as np

from dashboard.app import _format_metric_value


def test_format_metric_value_displays_vectors_as_readable_text():
    assert _format_metric_value([2.0, 1.0]) == "[2, 1]"
    assert _format_metric_value(np.array([1.41421356, 0.123456789])) == "[1.41421, 0.123457]"


def test_format_metric_value_keeps_scalar_metrics_supported():
    assert _format_metric_value(3) == 3
    assert _format_metric_value(np.float64(1.5)) == 1.5
