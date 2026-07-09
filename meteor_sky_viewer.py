r"""
meteor_sky_viewer.py

Simple GUI to display a meteor detection line on the celestial sphere.
- Center: North celestial pole (Dec = +90) at canvas center.
- Concentric circles: decreasing declination moving outward (Declination labeled).
- Default input file: C:\Users\kekke\Desktop\my_app\meteor\20251027_041015399_meteor_1_prob0.90_info.txt

Run:
    python meteor_sky_viewer.py [optional_path_to_info.txt]

This script opens a window (Tkinter) and does not save output files.
"""
from __future__ import annotations
import math
import os
import sys
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Optional, Tuple

DEFAULT_FILE = r"C:\Users\kekke\Desktop\my_app\meteor\20251027_041015399_meteor_1_prob0.90_info.txt"


def parse_info_file(path: str) -> Dict[str, str]:
    """Parse a simple key: value info file into a dict.
    Keys and values are stripped. Lines without ':' are ignored.
    """
    data: Dict[str, str] = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Info file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    return data


def get_float(data: Dict[str, str], key: str) -> Optional[float]:
    v = data.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> Tuple[float, float, float]:
    """Convert RA/Dec (in degrees) to a 3D unit vector on the celestial sphere.
    Returns (x, y, z) where:
    - x axis points toward RA=0, Dec=0
    - y axis points toward RA=90, Dec=0
    - z axis points toward North Celestial Pole (Dec=90)
    """
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    cos_dec = math.cos(dec_rad)
    x = cos_dec * math.cos(ra_rad)
    y = cos_dec * math.sin(ra_rad)
    z = math.sin(dec_rad)
    return x, y, z


def unit_vector_to_radec(x: float, y: float, z: float) -> Tuple[float, float]:
    """Convert a 3D unit vector back to RA/Dec in degrees."""
    dec_rad = math.asin(max(-1.0, min(1.0, z)))
    ra_rad = math.atan2(y, x)
    ra_deg = math.degrees(ra_rad)
    if ra_deg < 0:
        ra_deg += 360.0
    dec_deg = math.degrees(dec_rad)
    return ra_deg, dec_deg


def slerp(v1: Tuple[float, float, float], v2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    """Spherical linear interpolation between two unit vectors.
    t = 0 returns v1, t = 1 returns v2.
    """
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]))
    if dot > 0.9995:
        # vectors are nearly parallel, use linear interpolation and renormalize
        x = v1[0] + t * (v2[0] - v1[0])
        y = v1[1] + t * (v2[1] - v1[1])
        z = v1[2] + t * (v2[2] - v1[2])
        norm = math.sqrt(x*x + y*y + z*z)
        if norm < 1e-12:
            return v1
        return (x/norm, y/norm, z/norm)
    
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    a = math.sin((1.0 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    x = a * v1[0] + b * v2[0]
    y = a * v1[1] + b * v2[1]
    z = a * v1[2] + b * v2[2]
    return (x, y, z)


def sky_to_xy(ra_deg: float, dec_deg: float, cx: int, cy: int, pixel_per_deg: float) -> Tuple[float, float]:
    """Convert RA/Dec to screen x,y on a polar projection centered at North Celestial Pole.
    - radius is proportional to (90 - dec) degrees (so Dec=+90 -> r=0, Dec=-90 -> r=180)
    - angle: we set angle = 90 - RA (degrees) so RA=0 appears at top (y decreasing) and RA increases leftwards
    """
    r_deg = 90.0 - dec_deg
    angle_rad = math.radians(90.0 - ra_deg)
    r_px = r_deg * pixel_per_deg
    x = cx + r_px * math.cos(angle_rad)
    y = cy - r_px * math.sin(angle_rad)
    return x, y


def draw_sky(canvas: tk.Canvas, cx: int, cy: int, radius_px: int) -> None:
    """Draw concentric declination circles and RA meridians for context.
    Only shows northern hemisphere (Dec 0° to +90°).
    """
    # Draw declination circles every 10 deg from 90 down to 0 (northern hemisphere only)
    for dec in range(90, -10, -10):
        r_deg = 90 - dec
        r_px = r_deg * (radius_px / 90.0)  # radius_px now represents 90° range
        x0 = cx - r_px
        y0 = cy - r_px
        x1 = cx + r_px
        y1 = cy + r_px
        canvas.create_oval(x0, y0, x1, y1, outline="#666", width=1)
        # label at rightmost point
        if r_px >= 5:
            canvas.create_text(cx + r_px + 8, cy, text=f"{dec}°", anchor="w", fill="#333", font=("Arial", 9))

    # Draw RA meridians every 30 degrees (approx lines from pole outward)
    for ra in range(0, 360, 30):
        # draw a line from dec = +80 to dec = 0 to show meridian direction (northern hemisphere only)
        dec1 = 80.0
        dec2 = 0.0
        x1, y1 = sky_to_xy(ra, dec1, cx, cy, radius_px / 90.0)  # radius_px now represents 90° range
        x2, y2 = sky_to_xy(ra, dec2, cx, cy, radius_px / 90.0)
        canvas.create_line(x1, y1, x2, y2, fill="#999", dash=(3, 5))
        # RA label near the outer end (at Dec = -5° which is just outside visible range for context)
        label_x, label_y = sky_to_xy(ra, -5.0, cx, cy, radius_px / 90.0)
        canvas.create_text(label_x, label_y, text=f"{ra}°", fill="#444", font=("Arial", 8))


def draw_meteor(canvas: tk.Canvas, data: Dict[str, str], cx: int, cy: int, pixel_per_deg: float) -> None:
    ra_start = get_float(data, "RA Start (deg)")
    dec_start = get_float(data, "Dec Start (deg)")
    ra_end = get_float(data, "RA End (deg)")
    dec_end = get_float(data, "Dec End (deg)")

    if None in (ra_start, dec_start, ra_end, dec_end):
        raise ValueError("Missing RA/Dec start/end values in data")

    # Convert start and end to 3D unit vectors
    v_start = radec_to_unit_vector(ra_start, dec_start)
    v_end = radec_to_unit_vector(ra_end, dec_end)

    # Determine the outer circle radius (the sky boundary we want to clip to)
    # For northern hemisphere only: R represents 90° range (Dec 0° to +90°)
    R = (min(canvas.winfo_reqwidth(), canvas.winfo_reqheight()) // 2 - 60)
    if R <= 0:
        R = 90.0 * pixel_per_deg

    # Parameterize the great circle using angle phi around v_start
    # Compute orthonormal basis on the plane of the great circle
    dot = max(-1.0, min(1.0, v_start[0]*v_end[0] + v_start[1]*v_end[1] + v_start[2]*v_end[2]))
    theta = math.acos(dot)
    if abs(math.sin(theta)) > 1e-12:
        # u is unit vector perpendicular to v_start in the plane toward v_end
        u_raw = (v_end[0] - v_start[0]*dot, v_end[1] - v_start[1]*dot, v_end[2] - v_start[2]*dot)
        norm_u = math.sqrt(u_raw[0]*u_raw[0] + u_raw[1]*u_raw[1] + u_raw[2]*u_raw[2])
        u = (u_raw[0]/norm_u, u_raw[1]/norm_u, u_raw[2]/norm_u)
    else:
        # v_start and v_end are nearly the same or antipodal; pick arbitrary perpendicular u
        # choose a vector not parallel to v_start
        if abs(v_start[0]) < 0.9:
            tmp = (1.0, 0.0, 0.0)
        else:
            tmp = (0.0, 1.0, 0.0)
        # u = normalize(cross(tmp, v_start))
        ux = tmp[1]*v_start[2] - tmp[2]*v_start[1]
        uy = tmp[2]*v_start[0] - tmp[0]*v_start[2]
        uz = tmp[0]*v_start[1] - tmp[1]*v_start[0]
        norm_u = math.sqrt(ux*ux + uy*uy + uz*uz)
        if norm_u < 1e-12:
            u = (1.0, 0.0, 0.0)
        else:
            u = (ux/norm_u, uy/norm_u, uz/norm_u)

    # Sample phi across the full circle and build 2D projected points
    nphi = 2000
    phis = [(-math.pi) + (2.0*math.pi) * i / (nphi - 1) for i in range(nphi)]
    pts = []  # list of (x,y,d,phi)
    for phi in phis:
        # point on great circle: v = v_start*cos(phi) + u*sin(phi)
        vx = v_start[0]*math.cos(phi) + u[0]*math.sin(phi)
        vy = v_start[1]*math.cos(phi) + u[1]*math.sin(phi)
        vz = v_start[2]*math.cos(phi) + u[2]*math.sin(phi)
        # normalize to avoid drift
        normv = math.sqrt(vx*vx + vy*vy + vz*vz)
        if normv == 0:
            continue
        vx /= normv; vy /= normv; vz /= normv
        ra_p, dec_p = unit_vector_to_radec(vx, vy, vz)
        x, y = sky_to_xy(ra_p, dec_p, cx, cy, pixel_per_deg)
        d = math.hypot(x - cx, y - cy)
        pts.append((x, y, d, phi))

    # Find index in phis closest to phi=0 (which corresponds to v_start)
    # Because v_start corresponds to phi=0 by construction
    center_idx = nphi // 2

    # Build boolean array of inside/outside
    inside = [p[2] <= R for p in pts]

    # Find contiguous run of inside points that contains center_idx
    run_start = run_end = None
    if inside[center_idx]:
        # expand left
        i = center_idx
        while i >= 0 and inside[i]:
            i -= 1
        run_start = i + 1
        # expand right
        i = center_idx
        while i < len(inside) and inside[i]:
            i += 1
        run_end = i - 1
    else:
        # If center not inside, find nearest inside index to center
        closest = None
        best_dist = None
        for idx, val in enumerate(inside):
            if val:
                dist = abs(idx - center_idx)
                if best_dist is None or dist < best_dist:
                    best_dist = dist; closest = idx
        if closest is not None:
            # expand around closest
            i = closest
            while i >= 0 and inside[i]:
                i -= 1
            run_start = i + 1
            i = closest
            while i < len(inside) and inside[i]:
                i += 1
            run_end = i - 1

    line_coords = []
    if run_start is not None and run_end is not None and run_end >= run_start:
        # compute intersection at the left boundary if needed
        if run_start > 0 and not inside[run_start - 1]:
            x1, y1, d1, _ = pts[run_start - 1]
            x2, y2, d2, _ = pts[run_start]
            if abs(d2 - d1) > 1e-9:
                alpha = (R - d1) / (d2 - d1)
                x_cross = x1 + alpha * (x2 - x1)
                y_cross = y1 + alpha * (y2 - y1)
                line_coords.extend([x_cross, y_cross])
        # add inside points
        for idx in range(run_start, run_end + 1):
            x, y, d, _ = pts[idx]
            line_coords.extend([x, y])
        # compute intersection at the right boundary if needed
        if run_end < len(pts) - 1 and not inside[run_end + 1]:
            x1, y1, d1, _ = pts[run_end]
            x2, y2, d2, _ = pts[run_end + 1]
            if abs(d2 - d1) > 1e-9:
                alpha = (R - d1) / (d2 - d1)
                x_cross = x1 + alpha * (x2 - x1)
                y_cross = y1 + alpha * (y2 - y1)
                line_coords.extend([x_cross, y_cross])

    # Draw the great circle arc as a polyline if we have coordinates
    if len(line_coords) >= 4:
        canvas.create_line(line_coords, fill="red", width=3, smooth=False)
    
    # Draw original start and end points
    x1, y1 = sky_to_xy(ra_start, dec_start, cx, cy, pixel_per_deg)
    x2, y2 = sky_to_xy(ra_end, dec_end, cx, cy, pixel_per_deg)
    
    # endpoints
    r = 4
    canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="yellow", outline="#aa0")
    canvas.create_oval(x2 - r, y2 - r, x2 + r, y2 + r, fill="yellow", outline="#aa0")


def open_viewer(info_path: str) -> None:
    # parse
    try:
        data = parse_info_file(info_path)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to read info file:\n{e}")
        return

    # Setup Tkinter
    root = tk.Tk()
    root.title("Meteor Celestial Polar Viewer")

    width, height = 900, 900
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)

    cx, cy = width // 2, height // 2
    # Use up to radius_px for r_deg = 90 degrees (northern hemisphere only)
    radius_px = min(width, height) // 2 - 60
    pixel_per_deg = radius_px / 90.0

    # Draw sky grid
    draw_sky(canvas, cx, cy, radius_px)

    # Draw meteor; handle exceptions
    try:
        draw_meteor(canvas, data, cx, cy, pixel_per_deg)
    except Exception as e:
        messagebox.showerror("Plot error", f"Could not plot meteor: {e}")

    # Title and info text
    prob = data.get("Meteor Probability") or data.get("Meteor Probability:") or "?"
    canvas.create_text(10, 10, anchor="nw", text=f"File: {os.path.basename(info_path)}", fill="#111", font=("Arial", 10, "bold"))
    canvas.create_text(10, 28, anchor="nw", text=f"Meteor Probability: {prob}", fill="#111", font=("Arial", 10))
    canvas.create_text(10, 48, anchor="nw", text="Center: North Celestial Pole (Dec=+90). Concentric circles: Dec decreasing outward. Northern hemisphere only (Dec 0° to +90°).", fill="#333", font=("Arial", 9))

    # Close button
    btn = tk.Button(root, text="Close", command=root.destroy)
    btn.pack(pady=6)

    root.mainloop()


if __name__ == "__main__":
    path = DEFAULT_FILE
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"Info file not found: {path}")
        print("Provide a valid path as argument or update DEFAULT_FILE in the script.")
        sys.exit(1)
    open_viewer(path)
