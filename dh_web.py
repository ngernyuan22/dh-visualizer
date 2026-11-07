#!/usr/bin/env python3
"""
3-Link Denavit-Hartenberg Robot Visualizer - web app (Python / Dash + Plotly)

Standard (classical) DH convention:
    A_i = Rz(theta_i) . Tz(d_i) . Tx(r_i) . Rx(alpha_i)
    0T3 = A1 . A2 . A3

Install:  pip install dash numpy
Run:      python dh_web.py
Open:     http://127.0.0.1:8050
"""
import numpy as np
from dash import MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update

DEG = np.pi / 180.0

# Wolfram-demo defaults. Angles are stored in DEGREES (slider units).
DEFAULTS = [
    dict(type="R", theta=5 * 180 / 32, d=1.0, r=0.0, alpha=90.0),
    dict(type="R", theta=45.0,         d=0.0, r=1.0, alpha=0.0),
    dict(type="R", theta=-67.5,        d=0.0, r=1.0, alpha=0.0),
]
DEFAULT_OPEN = 0.2  # gripper opening (distance between the fingers)

KEYS = ("theta", "d", "r", "alpha")
ANGLE_KEYS = ("theta", "alpha")
GRIP_KEY = "open"
SYMS = {"theta": "θ", "d": "d", "r": "r", "alpha": "α", GRIP_KEY: "w"}
SPECS = {  # min, max, step
    "theta": (-180, 180, 1),
    "d": (-2.5, 2.5, 0.01),
    "r": (0.0, 2.5, 0.01),
    "alpha": (-180, 180, 1),
    GRIP_KEY: (0.0, 0.5, 0.01),
}
# every slider in the app, in a fixed order: (joint index, key). Gripper is "joint" 3.
ALL_SLIDERS = [(j, k) for j in range(3) for k in KEYS] + [(3, GRIP_KEY)]

AXIS_COLORS = ("#ff4d4d", "#3ddc84", "#4da3ff")  # x red, y green, z blue
REV_COLOR, PRIS_COLOR, GRIPPER_COLOR = "#dce6ff", "#ff9f43", "#2ec4b6"
MARKER_SIZE = 35  # joint markers: dot = revolute, square = prismatic (was 7)

# Screen-space (pixel) offsets of each label from the point it names. Because
# they are screen offsets, the labels stay apart while you orbit the view.
CALLOUT_OFFSETS = {0: (-95, 35), 1: (-105, -15), 2: (-95, -65), 3: (95, -70),
                   "gripper": (105, 25)}

BG, PANEL, BORDER, TEXT, MUTED = "#0b1020", "#121a2f", "#2a3658", "#eef3ff", "#9ba9c7"


def default_value(j, k):
    return DEFAULT_OPEN if k == GRIP_KEY else DEFAULTS[j][k]


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------
def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def tz(d):
    T = np.eye(4)
    T[2, 3] = d
    return T


def dh_matrix(theta, d, r, alpha):
    """Standard DH homogeneous transform (angles in radians)."""
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s * ca,  s * sa, r * c],
        [s,  c * ca, -c * sa, r * s],
        [0,       sa,      ca,     d],
        [0,        0,       0,     1.0],
    ])


def forward_kinematics(params):
    """params: list of dicts (angles in radians). Returns (links, T03)."""
    T = np.eye(4)
    links = []
    for p in params:
        T_after_d = T @ rz(p["theta"]) @ tz(p["d"])
        T_i = T @ dh_matrix(p["theta"], p["d"], p["r"], p["alpha"])
        links.append(dict(T_prev=T, T_after_d=T_after_d, T_i=T_i))
        T = T_i
    return links, T


# --------------------------------------------------------------------------
# Figure  (built as a plain dict of Python floats/lists: no NumPy arrays or
# plotly objects are handed to Dash, so it also works on setups where
# plotly's array encoder is broken)
# --------------------------------------------------------------------------
def _f(v):
    """numpy scalar/array -> plain python float (None stays None)."""
    return None if v is None else float(v)


def _line(a, b, color, width, dash="solid"):
    return dict(
        type="scatter3d", mode="lines",
        x=[_f(a[0]), _f(b[0])], y=[_f(a[1]), _f(b[1])], z=[_f(a[2]), _f(b[2])],
        line=dict(color=color, width=width, dash=dash),
        hoverinfo="skip", showlegend=False,
    )


def frame_traces(T, index, scale, axis_labels):
    """x/y/z arrows for one frame. Axis letters only if axis_labels is True."""
    o = T[:3, 3]
    traces = []
    for k, col in enumerate(AXIS_COLORS):
        direction = T[:3, k]
        tip = o + direction * scale
        traces.append(_line(o, tip, col, 6))
        traces.append(dict(
            type="cone",
            x=[_f(tip[0])], y=[_f(tip[1])], z=[_f(tip[2])],
            u=[_f(direction[0])], v=[_f(direction[1])], w=[_f(direction[2])],
            anchor="tip", sizemode="absolute", sizeref=0.12,
            colorscale=[[0, col], [1, col]], showscale=False,
            hoverinfo="skip", showlegend=False,
        ))
        if axis_labels:
            lp = o + direction * scale * 1.2
            traces.append(dict(
                type="scatter3d", mode="text",
                x=[_f(lp[0])], y=[_f(lp[1])], z=[_f(lp[2])],
                text=[f"{'xyz'[k]}{index}"], textfont=dict(color=col, size=11),
                hoverinfo="skip", showlegend=False,
            ))
    return traces


def callout(point, text, offset):
    """A label placed `offset` pixels away from `point`, joined to it by an arrow."""
    return dict(
        x=_f(point[0]), y=_f(point[1]), z=_f(point[2]),
        text=text, showarrow=True, ax=offset[0], ay=offset[1],
        arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#9fb3d9",
        font=dict(color="white", size=12), bgcolor="rgba(8,12,24,0.88)",
        bordercolor="#6c7fa8", borderwidth=1, borderpad=3,
    )


def gripper_geometry(T3, opening, approach="x"):
    """Two-finger gripper attached to frame 3.

    approach="x": the gripper points along x3, i.e. in line with the last link.
    approach="z": the gripper points along z3 (the standard DH tool direction).
    The fingers always open along y3. Returns (line segments, tip) in world coords.
    """
    a_idx = 0 if approach == "x" else 2
    origin, A, O = T3[:3, 3], T3[:3, a_idx], T3[:3, 1]
    half = opening / 2.0 + 0.015
    stem, finger, hook = 0.12, 0.22, 0.03
    z_tip = stem + finger

    def P(s, w):  # s = distance along the approach axis, w = along y3
        return origin + A * s + O * w

    segs = [
        (P(0, 0), P(stem, 0)),                          # wrist stem
        (P(stem, -half), P(stem, half)),                # palm bar
        (P(stem, -half), P(z_tip, -half)),              # finger A
        (P(stem, half), P(z_tip, half)),                # finger B
        (P(z_tip, -half), P(z_tip, -max(half - hook, 0))),  # hook A
        (P(z_tip, half), P(z_tip, max(half - hook, 0))),    # hook B
    ]
    return segs, P(z_tip, 0)


def make_figure(params_rad, types, opening, show_frames, show_labels,
                show_construction, show_grid, axis_sel, grip_axis="x"):
    links, T3 = forward_kinematics(params_rad)
    traces, annotations = [], []

    # ground grid on z = 0
    if show_grid:
        gx, gy, gz = [], [], []
        for v in np.arange(-3, 3.01, 0.5):
            v = float(v)
            gx += [v, v, None]; gy += [-3.0, 3.0, None]; gz += [0.0, 0.0, None]
            gx += [-3.0, 3.0, None]; gy += [v, v, None]; gz += [0.0, 0.0, None]
        traces.append(dict(
            type="scatter3d", mode="lines", x=gx, y=gy, z=gz,
            line=dict(color="#26314c", width=1),
            hoverinfo="skip", showlegend=False))

    # links (DH construction: d offset along z, then r offset along x)
    for L in links:
        a, b, c = L["T_prev"][:3, 3], L["T_after_d"][:3, 3], L["T_i"][:3, 3]
        if show_construction:
            if np.linalg.norm(b - a) > 1e-5:
                traces.append(_line(a, b, "#8aa0c8", 8))
            if np.linalg.norm(c - b) > 1e-5:
                traces.append(_line(b, c, "#f5c451", 10))
            traces.append(_line(a, c, "#53627f", 2, dash="dash"))
        else:
            traces.append(_line(a, c, "#f5c451", 10))

    # coordinate frames 0..3 + their callout labels
    frames = [(np.eye(4), 0, 0.56)] + [(L["T_i"], i + 1, 0.46)
                                       for i, L in enumerate(links)]
    for T, idx, scale in frames:
        if show_frames:
            letters = axis_sel == "all" or axis_sel == str(idx)
            traces += frame_traces(T, idx, scale, letters)
        if show_labels:
            annotations.append(callout(T[:3, 3], f"<b>Frame {idx}</b>",
                                       CALLOUT_OFFSETS[idx]))

    # joint markers: joint i sits on the z-axis of frame i-1.
    # dot = revolute, square = prismatic
    jp = [L["T_prev"][:3, 3] for L in links]
    traces.append(dict(
        type="scatter3d", mode="markers",
        x=[_f(p[0]) for p in jp], y=[_f(p[1]) for p in jp],
        z=[_f(p[2]) for p in jp],
        marker=dict(size=MARKER_SIZE,
                    symbol=["circle" if t == "R" else "square" for t in types],
                    color=[REV_COLOR if t == "R" else PRIS_COLOR for t in types]),
        hoverinfo="skip", showlegend=False))

    # gripper on frame 3
    segs, tip = gripper_geometry(T3, opening, grip_axis)
    for a, b in segs:
        traces.append(_line(a, b, GRIPPER_COLOR, 9))
    if show_labels:
        annotations.append(callout(tip, "<b>Gripper</b>", CALLOUT_OFFSETS["gripper"]))

    cx, cy, cz, h = 0.6, 0.0, 0.6, 2.4
    axis = dict(
        backgroundcolor="#0a1122", gridcolor="#26314c", showbackground=True,
        zerolinecolor="#47567a", color="#8d9bbb",
    )
    layout = dict(
        paper_bgcolor="#080d19",
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",  # keeps camera/zoom while sliders move
        scene=dict(
            xaxis=dict(range=[cx - h, cx + h], title=dict(text="X"), **axis),
            yaxis=dict(range=[cy - h, cy + h], title=dict(text="Y"), **axis),
            zaxis=dict(range=[cz - h, cz + h], title=dict(text="Z"), **axis),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.3, z=0.9)),
            annotations=annotations,
        ),
    )
    return dict(data=traces, layout=layout), T3


def matrix_table(T):
    def fmt(v):
        return "0" if abs(v) < 1e-10 else f"{v:.3f}"
    return html.Table(
        [html.Tr([html.Td(fmt(v), style=dict(
            padding="3px 6px", textAlign="right",
            borderBottom="1px solid rgba(70,85,120,.3)")) for v in row])
         for row in T],
        style=dict(width="100%", borderCollapse="collapse", fontSize="12px",
                   fontFamily="ui-monospace, Menlo, Consolas, monospace",
                   color="#cfe0ff"),
    )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
card_style = dict(border=f"1px solid {BORDER}", background="rgba(24,34,61,.72)",
                  borderRadius="12px", padding="12px", marginBottom="10px")
input_style = dict(width="84px", background="#0e162b", color=TEXT,
                   border="1px solid #344366", borderRadius="7px",
                   padding="5px 6px", fontSize="13px", textAlign="right")


def sl_id(j, k):
    return {"type": "sl", "j": j, "k": k}


def param_row(j, k):
    """slider + exact-value box (+ the same angle in the other unit)."""
    lo, hi, step = SPECS[k]
    v0 = default_value(j, k)
    return html.Div(style=dict(margin="6px 0"), children=[
        html.Div(
            style=dict(display="grid", gridTemplateColumns="34px 1fr 84px",
                       gap="8px", alignItems="center"),
            children=[
                html.Span(f"{SYMS[k]}{j + 1 if j < 3 else ''}",
                          id={"type": "lab", "j": j, "k": k},
                          style=dict(fontSize="13px", color="#c7d3ed")),
                dcc.Slider(min=lo, max=hi, step=step, value=v0, id=sl_id(j, k),
                           marks=None, updatemode="drag"),
                dcc.Input(id={"type": "num", "j": j, "k": k}, type="number",
                          value=round(v0, 4), debounce=True, step="any",
                          style=input_style),
            ]),
        html.Div(id={"type": "alt", "j": j, "k": k},
                 style=dict(textAlign="right", fontSize="11px", color=MUTED,
                            height="13px", marginRight="2px")),
    ])


def joint_card(j):
    return html.Div(style=card_style, children=[
        html.Div(style=dict(display="flex", justifyContent="space-between",
                            alignItems="center", marginBottom="6px"),
                 children=[
            html.Div(f"Joint {j + 1}", style=dict(fontWeight=700, fontSize="14px")),
            dcc.Dropdown(
                options=[{"label": "Revolute (dot)", "value": "R"},
                         {"label": "Prismatic (square)", "value": "P"}],
                value=DEFAULTS[j]["type"], id=f"type{j}", clearable=False,
                style=dict(width="170px", color="#000", fontSize="13px")),
        ]),
        *[param_row(j, k) for k in KEYS],
    ])


app = Dash(__name__)
app.title = "3-Link DH Robot Visualizer"
server = app.server  # web hosts (gunicorn dh_web:server) use this; local runs ignore it

app.layout = html.Div(
    style=dict(display="grid", gridTemplateColumns="430px 1fr", height="100vh",
               background=BG, color=TEXT,
               fontFamily="Inter, system-ui, -apple-system, Segoe UI, sans-serif"),
    children=[
        html.Div(
            style=dict(overflowY="auto", padding="18px",
                       background=f"linear-gradient(180deg,{PANEL},#0f1628)",
                       borderRight=f"1px solid {BORDER}"),
            children=[
                html.H1("3-Link DH Robot Visualizer",
                        style=dict(fontSize="19px", margin="0 0 7px")),
                html.P("Interactive standard Denavit–Hartenberg model. Drag a "
                       "slider or type an exact value and press Enter. Frame "
                       "axes: x red, y green, z blue.",
                       style=dict(color=MUTED, fontSize="13px", lineHeight="1.45")),

                html.Div(style=dict(display="flex", alignItems="center",
                                    gap="12px", margin="14px 0 4px"), children=[
                    html.Span("Angle units:", style=dict(fontSize="13px",
                                                         color="#c7d3ed")),
                    dcc.RadioItems(
                        id="angle_mode",
                        options=[{"label": " Degrees", "value": "deg"},
                                 {"label": " Radians", "value": "rad"}],
                        value="deg",
                        labelStyle=dict(display="inline-block", marginRight="14px",
                                        fontSize="13px", color=TEXT)),
                ]),

                html.H2("Joint / link parameters",
                        style=dict(fontSize="14px", margin="14px 0 9px")),
                *[joint_card(j) for j in range(3)],

                html.Div(style=card_style, children=[
                    html.Div("Gripper", style=dict(fontWeight=700, fontSize="14px",
                                                   marginBottom="4px")),
                    html.Div("w = opening between the fingers (along y₃)",
                             style=dict(fontSize="11px", color=MUTED)),
                    param_row(3, GRIP_KEY),
                    html.Div("Gripper direction:", style=dict(
                        fontSize="12px", color="#c7d3ed", margin="8px 0 2px")),
                    dcc.RadioItems(
                        id="grip_axis",
                        options=[{"label": " In line with link 3 (along x₃)", "value": "x"},
                                 {"label": " Along z₃ (standard DH tool axis)", "value": "z"}],
                        value="x",
                        labelStyle=dict(display="block", margin="3px 0",
                                        fontSize="12px", color=TEXT)),
                ]),

                html.Div(style=card_style, children=[
                    html.Div("Display", style=dict(fontWeight=700, fontSize="13px",
                                                   marginBottom="6px")),
                    dcc.Checklist(
                        id="display",
                        options=[
                            {"label": " Show DH coordinate frames", "value": "frames"},
                            {"label": " Show frame labels (callouts with arrows)", "value": "labels"},
                            {"label": " Show DH offset construction", "value": "construction"},
                            {"label": " Show ground grid", "value": "grid"},
                        ],
                        value=["frames", "labels", "construction", "grid"],
                        labelStyle=dict(display="block", margin="5px 0",
                                        fontSize="13px", color="#c7d3ed"),
                    ),
                    html.Div("Axis letters (x, y, z) on:",
                             style=dict(fontSize="12px", color="#c7d3ed",
                                        margin="10px 0 4px")),
                    dcc.Dropdown(
                        id="axis_labels", clearable=False, value="none",
                        options=[{"label": "None", "value": "none"},
                                 {"label": "All frames", "value": "all"},
                                 {"label": "Frame 0 only", "value": "0"},
                                 {"label": "Frame 1 only", "value": "1"},
                                 {"label": "Frame 2 only", "value": "2"},
                                 {"label": "Frame 3 only", "value": "3"}],
                        style=dict(color="#000", fontSize="13px")),
                    html.Div(style=dict(fontSize="12px", margin="12px 0 2px",
                                        color="#c7d3ed"), children=[
                        html.Span("● ", style=dict(color=REV_COLOR)), "revolute joint    ",
                        html.Span("■ ", style=dict(color=PRIS_COLOR)), "prismatic joint",
                    ]),
                    html.Div(style=dict(display="grid",
                                        gridTemplateColumns="1fr 1fr",
                                        gap="8px", marginTop="12px"),
                             children=[
                        html.Button("Reset Wolfram example", id="reset", n_clicks=0),
                        html.Button("Zero angles", id="zero", n_clicks=0),
                    ]),
                ]),

                html.H2("Standard DH transform",
                        style=dict(fontSize="14px", margin="16px 0 9px")),
                html.Pre(
                    "Aᵢ = Rz(θᵢ) · Tz(dᵢ) · Tx(rᵢ) · Rx(αᵢ)\n\n"
                    "[ cθ  −sθcα   sθsα   r cθ ]\n"
                    "[ sθ   cθcα  −cθsα   r sθ ]\n"
                    "[  0     sα      cα      d ]\n"
                    "[  0      0       0      1 ]",
                    style=dict(fontSize="11px", lineHeight="1.45", color="#cfe0ff",
                               background="#0b1326", borderRadius="8px",
                               padding="9px", overflowX="auto")),
                html.H2("End-effector transform ⁰T₃",
                        style=dict(fontSize="14px", margin="16px 0 9px")),
                html.Div(id="matrix", style=card_style),
            ],
        ),
        dcc.Graph(id="graph", style=dict(height="100vh"),
                  config=dict(displaylogo=False)),
    ],
)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
# 1) Keep each slider and its number box in sync (and show the other unit).
#    The slider always stores DEGREES for angles; the box shows the chosen unit.
def register_sync(key):
    lo, hi, _ = SPECS[key]
    is_angle = key in ANGLE_KEYS

    def shown(sl_val, mode):
        v = sl_val * DEG if (is_angle and mode == "rad") else sl_val
        return round(float(v), 4)

    sl = {"type": "sl", "j": MATCH, "k": key}
    num = {"type": "num", "j": MATCH, "k": key}
    alt = {"type": "alt", "j": MATCH, "k": key}

    @app.callback(Output(num, "value"), Output(alt, "children"),
                  Input(sl, "value"), Input("angle_mode", "value"),
                  State(num, "value"))
    def slider_to_box(sl_val, mode, cur):
        if sl_val is None:
            return no_update, no_update
        text = ""
        if is_angle:
            text = (f"= {sl_val * DEG:.4f} rad" if mode == "deg"
                    else f"= {sl_val:.2f}°")
        new = shown(sl_val, mode)
        if cur is not None and abs(cur - new) < 1e-9:
            return no_update, text
        return new, text

    @app.callback(Output(sl, "value"), Input(num, "value"),
                  State(sl, "value"), State("angle_mode", "value"),
                  prevent_initial_call=True)
    def box_to_slider(typed, cur_sl, mode):
        if typed is None:
            return no_update
        if cur_sl is not None and abs(typed - shown(cur_sl, mode)) < 1e-9:
            return no_update  # box just echoed the slider
        deg = typed / DEG if (is_angle and mode == "rad") else typed
        return min(max(deg, lo), hi)  # keep within the slider's range


for _key in KEYS + (GRIP_KEY,):
    register_sync(_key)


# 2) Redraw the 3D scene whenever anything changes.
@app.callback(
    [Output("graph", "figure"), Output("matrix", "children")]
    + [Output({"type": "lab", "j": j, "k": k}, "style")
       for j in range(3) for k in KEYS],
    [Input(sl_id(j, k), "value") for j, k in ALL_SLIDERS]
    + [Input(f"type{j}", "value") for j in range(3)]
    + [Input("display", "value"), Input("axis_labels", "value"),
       Input("grip_axis", "value")],
)
def update(*args):
    slider_vals = [v if v is not None else default_value(j, k)
                   for v, (j, k) in zip(args[:13], ALL_SLIDERS)]
    types = args[13:16]
    display = args[16] or []
    axis_sel = args[17] or "none"
    grip_axis = args[18] or "x"

    params_rad, lab_styles = [], []
    for j in range(3):
        row = {}
        for n, key in enumerate(KEYS):
            v = slider_vals[j * 4 + n]
            row[key] = v * DEG if key in ANGLE_KEYS else v
            var = "theta" if types[j] == "R" else "d"   # the joint variable
            hl = key == var
            lab_styles.append(dict(fontSize="13px",
                                   color="#ffd66b" if hl else "#c7d3ed",
                                   fontWeight=800 if hl else 400))
        params_rad.append(row)

    fig, T3 = make_figure(
        params_rad, types, slider_vals[12],
        show_frames="frames" in display,
        show_labels="labels" in display,
        show_construction="construction" in display,
        show_grid="grid" in display,
        axis_sel=axis_sel,
        grip_axis=grip_axis,
    )
    return [fig, matrix_table(T3), *lab_styles]


# 3) Reset / zero buttons write to the sliders (the sync callbacks then update
#    the number boxes).
@app.callback(
    [Output(sl_id(j, k), "value", allow_duplicate=True) for j, k in ALL_SLIDERS]
    + [Output(f"type{j}", "value") for j in range(3)],
    Input("reset", "n_clicks"), Input("zero", "n_clicks"),
    prevent_initial_call=True,
)
def reset_or_zero(_r, _z):
    if ctx.triggered_id == "reset":
        return ([default_value(j, k) for j, k in ALL_SLIDERS]
                + [DEFAULTS[j]["type"] for j in range(3)])
    # zero: only the theta sliders change
    return ([0 if k == "theta" else no_update for j, k in ALL_SLIDERS]
            + [no_update] * 3)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
