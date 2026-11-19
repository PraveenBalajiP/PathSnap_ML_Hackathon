ate_rank","candidate_distance_m","projection_fraction","arc_length_m","from_node_out_degree","from_node_in_degree","to_node_out_degree","to_node_in_degree"]
ANGLES=["gps_heading_deg","gps_heading_change_deg","arc_bearing_deg","heading_difference_deg"]
SEED=42

def features(df):
    x=df[NUMERIC].copy()
    for c in ANGLES:
        r=np.deg2rad(pd.to_numeric(df[c],errors="coerce"));