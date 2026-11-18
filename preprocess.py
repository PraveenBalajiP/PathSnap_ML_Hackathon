"""
PathSnap preprocessing: loading, projection, arc cleaning, candidate
generation and reference-label assignment for ONE trip.

Design notes (be ready to explain these in the viva):
  * arc_id is the 0-based ROW INDEX of the .arcs file, so we never drop
    rows from the arcs array; bad arcs are masked instead.
  * All geometry is done in UTM zone 32N (EPSG:32632, metres). The data
    lies at lon ~8.6E, which is inside zone 32 (6E-12E).
  * Labels: .route is a sequence of arcs, not one arc per GPS point, so we
    assign each point to a route arc with a monotone (order-preserving)
    dynamic program that minimises total point-to-arc distance.
"""
import os
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree, linestrings, points as shp_points

CRS_UTM = "EPSG:32632"
_tf = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)


# ----------------------------------------------------------------- loading
def load_trip(folder, trip_id, need_route=True):
    """trip_id like '00000001'. Returns dict of raw arrays."""
    def p(ext):
        nested = os.path.join(folder, trip_id, f"{trip_id}.{ext}")
        flat = os.path.join(folder, f"{trip_id}.{ext}")
        return nested if os.path.exists(nested) else flat
    track = np.loadtxt(p("track"))                  # lon, lat, t
    nodes = np.loadtxt(p("nodes"))                  # lon, lat
    arcs = np.loadtxt(p("arcs"), dtype=np.int64)    # from, to
    route = None
    if need_route and os.path.exists(p("route")):
        route = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    return dict(track=track, nodes=nodes, arcs=arcs, route=route)


def project(lonlat):
    x, y = _tf.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


# ---------------------------------------------------------------- cleaning
def validate_track(track):
    """Return (clean_track_mask, report). Flags NaN, out-of-range coords,
    non-increasing timestamps. Does NOT interpolate or drop silently."""
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    bad_nan = ~np.isfinite(track).all(axis=1)
    bad_rng = (np.abs(lon) > 180) | (np.abs(lat) > 90)
    dt = np.diff(t, prepend=t[0] - 1)
    bad_time = dt <= 0
    keep = ~(bad_nan | bad_rng | bad_time)
    rep = dict(n=len(track), nan=int(bad_nan.sum()), out_of_range=int(bad_rng.sum()),
               non_increasing_time=int(bad_time.sum()),
               max_gap_s=float(np.max(np.diff(t))) if len(t) > 1 else 0.0)
    return keep, rep


def build_network(nodes_ll, arcs):
    """Project nodes, build arc geometry, mask degenerate arcs."""
    nxy = project(nodes_ll)
    a, b = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    length = np.hypot(*(b - a).T)
    valid = length > 0.5                     # drop zero / sub-0.5 m arcs from the index
    bearing = np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360  # 0=N, clockwise
    return dict(nxy=nxy, a=a, b=b, length=length, valid=valid, bearing=bearing, arcs=arcs)


def build_index(net):
    ids = np.flatnonzero(net["valid"])
    geoms = linestrings(np.stack([net["a"][ids], net["b"][ids]], axis=1))
    return STRtree(geoms), ids, geoms


# ------------------------------------------------------ geometry helpers
def seg_dist_frac(p, a, b):
    """Distance from points p to segments a-b (all (n,2)) and fraction along."""
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    f = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
    proj = a + f[:, None] * ab
    return np.hypot(*(p - proj).T), f


def gps_bearing_speed(xy, t):
    """Heading (deg, 0=N) and speed (m/s) per point from central differences."""
    n = len(xy)
    nxt, prv = np.minimum(np.arange(n) + 1, n - 1), np.maximum(np.arange(n) - 1, 0)
    d = xy[nxt] - xy[prv]
    dt = np.maximum(t[nxt] - t[prv], 1e-6)
    spd = np.hypot(*d.T) / dt
    brg = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    return brg, spd


# -------------------------------------------------- candidate generation
def generate_candidates(xy, t, net, tree, ids, radius=100.0):
    """All arcs within `radius` m of each GPS point -> DataFrame, one row per
    (point, arc) pair. Contains only raw geometric quantities; richer
    feature engineering (Person 2) builds on top of this."""
    pts = shp_points(xy)
    pi, gi = tree.query(pts, predicate="dwithin", distance=radius)
    arc = ids[gi]
    dist, frac = seg_dist_frac(xy[pi], net["a"][arc], net["b"][arc])
    brg, spd = gps_bearing_speed(xy, t)
    dh = np.abs((brg[pi] - net["bearing"][arc] + 180) % 360 - 180)
    df = pd.DataFrame(dict(pt=pi, arc_id=arc, dist=dist, frac=frac,
                           arc_len=net["length"][arc], arc_bearing=net["bearing"][arc],
                           gps_bearing=brg[pi], head_diff=dh, speed=spd[pi]))
    return df.sort_values(["pt", "dist"], kind="stable").reset_index(drop=True)


# -------------------------------------------------------------- labelling
def assign_reference_arcs(xy, route, net):
    """Monotone assignment of GPS points to reference route arcs.
    dp[i, j] = d(i, route[j]) + min_{k<=j} dp[i-1, k]   (order preserved)
    Returns (label_arc_id per point, distance to that arc)."""
    a, b = net["a"][route], net["b"][route]
    n, m = len(xy), len(route)
    D = np.empty((n, m))
    for j in range(m):                                   # m ~ few hundred
        D[:, j], _ = seg_dist_frac(xy, np.broadcast_to(a[j], xy.shape), np.broadcast_to(b[j], xy.shape))
    dp = np.empty((n, m)); bp = np.empty((n, m), dtype=np.int32)
    dp[0] = D[0]
    for i in range(1, n):
        prev = dp[i - 1]
        cm = np.minimum.accumulate(prev)                 # best cost over k<=j
        # argmin index of running min
        idx = np.arange(m)
        is_new = np.r_[True, prev[1:] < cm[:-1]]
        am = np.maximum.accumulate(np.where(is_new, idx, 0))
        dp[i] = D[i] + cm
        bp[i] = am
    j = int(np.argmin(dp[-1])); path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = bp[i, j]
    return route[path], D[np.arange(n), path], path


def label_candidates(cands, label_arc):
    cands["label"] = (cands["arc_id"].values == label_arc[cands["pt"].values]).astype(np.int8)
    return cands


# ------------------------------------------------------------ one-call API
def process_trip(folder, trip_id, radius=100.0, with_labels=True):
    raw = load_trip(folder, trip_id, need_route=with_labels)
    keep, rep = validate_track(raw["track"])
    track = raw["track"][keep]
    xy, t = project(track[:, :2]), track[:, 2]
    net = build_network(raw["nodes"], raw["arcs"])
    tree, ids, _ = build_index(net)
    cands = generate_candidates(xy, t, net, tree, ids, radius)
    out = dict(track=track, xy=xy, t=t, net=net, cands=cands, report=rep, route=raw["route"])
    if with_labels and raw["route"] is not None:
        lab, ldist, path = assign_reference_arcs(xy, raw["route"], net)
        cands = label_candidates(cands, lab)
        # per-point diagnostics: is the reference arc inside the candidate set?
        hit = cands.groupby("pt")["label"].max().reindex(range(len(xy)), fill_value=0)
        out.update(label_arc=lab, label_dist=ldist, route_pos=path, point_has_true_candidate=hit.values.astype(bool))
        out["cands"] = cands
    return out
