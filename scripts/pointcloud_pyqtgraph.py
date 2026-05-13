from __future__ import annotations

import numpy as np


def range_center(value_range: tuple[float, float]) -> float:
    return (value_range[0] + value_range[1]) * 0.5


def range_span(value_range: tuple[float, float]) -> float:
    return max(value_range[1] - value_range[0], 1e-6)


def axis_reference(value_range: tuple[float, float], preferred: float = 0.0) -> float:
    return float(np.clip(preferred, value_range[0], value_range[1]))


def tick_values(value_range: tuple[float, float]) -> np.ndarray:
    span = range_span(value_range)
    if span <= 2.0:
        step = 0.5
    elif span <= 5.0:
        step = 1.0
    else:
        step = 2.0

    start = np.ceil(value_range[0] / step) * step
    stop = np.floor(value_range[1] / step) * step
    values = np.arange(start, stop + step * 0.5, step)
    return values[(values >= value_range[0] - 1e-9) & (values <= value_range[1] + 1e-9)]


def add_gl_line(view, gl, points, color, width: float = 1.0, mode: str = 'lines'):
    item = gl.GLLinePlotItem(
        pos=np.asarray(points, dtype=np.float32),
        color=color,
        width=width,
        mode=mode,
        antialias=True,
    )
    view.addItem(item)
    return item


def add_gl_text(view, gl, qt_gui, pos, text: str, color, size: int = 10):
    font = qt_gui.QFont('Helvetica', size)
    item = gl.GLTextItem(
        pos=np.asarray(pos, dtype=np.float32),
        text=text,
        color=color,
        font=font,
    )
    view.addItem(item)
    return item


def add_axis_guides(view, gl, qt_gui, display_ranges) -> None:
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
    x_ref = axis_reference((x_min, x_max))
    y_ref = axis_reference((y_min, y_max))
    z_ref = axis_reference((z_min, z_max))
    x_span = range_span((x_min, x_max))
    y_span = range_span((y_min, y_max))
    z_span = range_span((z_min, z_max))
    tick_len = max(min(x_span, y_span, z_span) * 0.035, 0.04)
    axis_width = 2.5
    tick_width = 1.6
    box_color = (0.55, 0.55, 0.55, 0.45)
    tick_color = (0.9, 0.9, 0.9, 0.9)
    x_color = (1.0, 0.25, 0.25, 1.0)
    y_color = (0.25, 1.0, 0.35, 1.0)
    z_color = (0.3, 0.55, 1.0, 1.0)

    corners = [
        (x_min, y_min, z_min), (x_max, y_min, z_min),
        (x_min, y_max, z_min), (x_max, y_max, z_min),
        (x_min, y_min, z_max), (x_max, y_min, z_max),
        (x_min, y_max, z_max), (x_max, y_max, z_max),
    ]
    edges = [
        (corners[0], corners[1]), (corners[2], corners[3]),
        (corners[4], corners[5]), (corners[6], corners[7]),
        (corners[0], corners[2]), (corners[1], corners[3]),
        (corners[4], corners[6]), (corners[5], corners[7]),
        (corners[0], corners[4]), (corners[1], corners[5]),
        (corners[2], corners[6]), (corners[3], corners[7]),
    ]
    add_gl_line(view, gl, [point for edge in edges for point in edge], box_color, width=1.0)

    add_gl_line(view, gl, [(x_min, y_ref, z_ref), (x_max, y_ref, z_ref)], x_color, width=axis_width)
    add_gl_line(view, gl, [(x_ref, y_min, z_ref), (x_ref, y_max, z_ref)], y_color, width=axis_width)
    add_gl_line(view, gl, [(x_ref, y_ref, z_min), (x_ref, y_ref, z_max)], z_color, width=axis_width)

    tick_segments = []
    for x in tick_values((x_min, x_max)):
        tick_segments.extend([(x, y_ref - tick_len, z_ref), (x, y_ref + tick_len, z_ref)])
    for y in tick_values((y_min, y_max)):
        tick_segments.extend([(x_ref - tick_len, y, z_ref), (x_ref + tick_len, y, z_ref)])
    for z in tick_values((z_min, z_max)):
        tick_segments.extend([(x_ref - tick_len, y_ref, z), (x_ref + tick_len, y_ref, z)])
    add_gl_line(view, gl, tick_segments, tick_color, width=tick_width)

    for x in tick_values((x_min, x_max)):
        add_gl_text(view, gl, qt_gui, (x, y_ref - tick_len * 4.0, z_ref - tick_len * 2.0), f'{x:g}', tick_color)
    for y in tick_values((y_min, y_max)):
        add_gl_text(view, gl, qt_gui, (x_ref - tick_len * 4.5, y, z_ref - tick_len * 2.0), f'{y:g}', tick_color)
    for z in tick_values((z_min, z_max)):
        add_gl_text(view, gl, qt_gui, (x_ref - tick_len * 5.0, y_ref - tick_len * 2.0, z), f'{z:g}', tick_color)

    add_gl_text(view, gl, qt_gui, (x_max + x_span * 0.06, y_ref, z_ref), 'X horiz (m)', x_color, size=12)
    add_gl_text(view, gl, qt_gui, (x_ref, y_max + y_span * 0.06, z_ref), 'Y depth (m)', y_color, size=12)
    add_gl_text(view, gl, qt_gui, (x_ref, y_ref, z_max + z_span * 0.08), 'Z height (m)', z_color, size=12)


def init_plot(display_ranges, title: str = 'Realtime points 3D (pyqtgraph OpenGL)'):
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtGui
        from pyqtgraph.Qt import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            '需要先安裝 pyqtgraph 3D 依賴：python -m pip install pyqtgraph PyQt5 PyOpenGL'
        ) from exc

    app = QtWidgets.QApplication.instance() or pg.mkQApp('MARS UART 3D Point Cloud')
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
    x_center = range_center((x_min, x_max))
    y_center = range_center((y_min, y_max))
    z_center = range_center((z_min, z_max))
    x_span = range_span((x_min, x_max))
    y_span = range_span((y_min, y_max))
    z_span = range_span((z_min, z_max))

    view = gl.GLViewWidget()
    view.setWindowTitle(title)
    view.setBackgroundColor((18, 20, 24))
    view.setCameraPosition(distance=max(x_span, y_span, z_span) * 1.8, elevation=20, azimuth=-62)
    view.opts['center'].setX(x_center)
    view.opts['center'].setY(y_center)
    view.opts['center'].setZ(z_center)

    grid = gl.GLGridItem()
    grid.setSize(x=x_span, y=y_span)
    grid.setSpacing(x=max(x_span / 8.0, 0.25), y=max(y_span / 8.0, 0.25))
    grid.translate(x_center, y_center, z_min)
    view.addItem(grid)

    add_axis_guides(view, gl, QtGui, display_ranges)

    scatter = gl.GLScatterPlotItem(
        pos=np.empty((0, 3), dtype=np.float32),
        color=np.empty((0, 4), dtype=np.float32),
        size=8,
        pxMode=True,
    )
    view.addItem(scatter)
    view.show()
    app.processEvents()
    return {'app': app, 'view': view, 'scatter': scatter}


def clip_display_points(points: np.ndarray, display_ranges) -> np.ndarray:
    if points is None or points.size == 0:
        return np.zeros((0, 5), dtype=np.float64)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = display_ranges
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def y_to_rgba(y_values: np.ndarray, y_range: tuple[float, float]) -> np.ndarray:
    if y_values.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    y_min, _y_max = y_range
    t = np.clip((y_values - y_min) / range_span(y_range), 0.0, 1.0).astype(np.float32)
    return np.column_stack((
        0.2 + 0.7 * t,
        0.9 - 0.5 * t,
        1.0 - 0.8 * t,
        np.ones_like(t),
    )).astype(np.float32)


def update_plot_points(points: np.ndarray, plot, display_ranges) -> None:
    points = clip_display_points(points, display_ranges)
    scatter = plot['scatter']
    if points.size == 0:
        scatter.setData(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
        )
    else:
        scatter.setData(
            pos=points[:, 0:3].astype(np.float32, copy=False),
            color=y_to_rgba(points[:, 1], display_ranges[1]),
            size=8,
            pxMode=True,
        )
    plot['app'].processEvents()


def process_plot_events(plot) -> None:
    plot['app'].processEvents()
