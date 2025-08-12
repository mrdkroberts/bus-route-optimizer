
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import networkx as nx
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import io
import requests
import urllib.parse

st.set_page_config(page_title="Dawson's Bus Tool", page_icon="🚌", layout="wide")
st.title("Dawson's Bus Tool 🚌")
st.write("Upload students clustered into bus stops, enter a school address, and generate an optimized route, map, and driver sheet.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    use_turn_by_turn = st.checkbox("Fetch turn‑by‑turn directions (OSRM)", value=False, help="Uses free OSRM API per leg; may be rate‑limited.")
    average_speed_mph = st.number_input("Cruise speed for ETA (mph)", min_value=5, max_value=55, value=22)
    distance_units = st.selectbox("Distance units", ["miles", "km"], index=0)
    st.markdown("---")
    st.caption("CSV required columns: either **name, stop_address** or **name, stop_id, stop_address**. Extra columns are okay.")

# File upload & addresses
col1, col2 = st.columns(2)
with col1:
    start_address = st.text_input("Starting location (full)", placeholder="e.g., 1 Depot Rd, Your City, State")
with col2:
    school_address = st.text_input("Destination school address (full)", placeholder="e.g., 123 School Ln, Your City, State")

uploaded = st.file_uploader("Upload CSV of students grouped into stops", type=["csv"])


# Helpers
def geocode_address(geocoder, address: str):
    if not address or not isinstance(address, str):
        return None
    try:
        loc = geocoder.geocode(address)
        if not loc:
            return None
        return (loc.latitude, loc.longitude)
    except Exception:
        return None

def osrm_turn_by_turn(lat1, lon1, lat2, lon2):
    base_url = "http://router.project-osrm.org/route/v1/driving/"
    coords = f"{lon1},{lat1};{lon2},{lat2}"
    params = "?overview=false&steps=true&geometries=geojson"
    url = base_url + urllib.parse.quote(coords, safe=";,") + params
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "routes" not in data or not data["routes"]:
            return "", None, None
        route_info = data["routes"][0]
        distance_m = float(route_info.get("distance", 0.0))
        duration_s = float(route_info.get("duration", 0.0))
        steps = []
        for leg in route_info.get("legs", []):
            for step in leg.get("steps", []):
                instr = step.get("maneuver", {}).get("instruction", "")
                dist = step.get("distance", 0.0)
                steps.append(f"{instr} ({dist:.0f} m)")
        return " | ".join(steps), distance_m, duration_s
    except Exception:
        return "", None, None

def miles_km(value_meters, units):
    if units == "miles":
        return value_meters / 1609.344
    else:
        return value_meters / 1000.0

def fmt_dist(value, units):
    if value is None:
        return ""
    if units == "miles":
        return f"{value:.2f} mi"
    else:
        return f"{value:.2f} km"

if uploaded is not None and school_address:
    # Load
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Validate columns
    cols = [c.lower() for c in df.columns]
    if "name" not in cols or "stop_address" not in cols:
        st.error("CSV must include at least 'name' and 'stop_address' columns (case-insensitive). Optional: 'stop_id'.")
        st.stop()

    # Normalize columns
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    if "stop_id" not in df.columns:
        # create synthetic stop_id from stop_address
        df["stop_id"] = df["stop_address"].astype("category").cat.codes

    # Geocoder (Nominatim) with polite rate limit
    geolocator = Nominatim(user_agent="dawsons_bus_tool")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

    # Geocode school
    with st.spinner("Geocoding school address..."):
        school_coords = geocode_address(geolocator, school_address)
    if not school_coords:
        st.error("Could not geocode the school address. Please check the address and try again.")
        st.stop()

    # Unique stops
    stops = df[["stop_id", "stop_address"]].drop_duplicates().sort_values("stop_id").reset_index(drop=True)

    # Geocode stops (cache results within the session)
    @st.cache_data(show_spinner=False)
    def geocode_stops(stops_df):
        results = []
        for _, row in stops_df.iterrows():
            coords = geocode_address(geolocator, row["stop_address"])
            results.append((row["stop_id"], row["stop_address"], coords))
        out = pd.DataFrame(results, columns=["stop_id", "stop_address", "coords"])
        return out

    with st.spinner("Geocoding bus stops..."):
        geo_stops = geocode_stops(stops)

    # Drop any failed geocodes
    bad = geo_stops[geo_stops["coords"].isna()]
    if not bad.empty:
        st.warning(f"{len(bad)} stop address(es) could not be geocoded and will be skipped:\n" +
                   "\n".join(bad["stop_address"].tolist()))
    geo_stops = geo_stops.dropna(subset=["coords"]).copy()
    if geo_stops.empty:
        st.error("No stops could be geocoded. Please verify addresses and try again.")
        st.stop()

    # Build distance graph (for TSP heuristic using great-circle distances)
    coords_dict = {"School": school_coords}
    for _, r in geo_stops.iterrows():
        coords_dict[str(r["stop_id"])] = r["coords"]

    # Complete graph
    G = nx.complete_graph(len(coords_dict))
    mapping = {i: list(coords_dict.keys())[i] for i in range(len(coords_dict))}
    G = nx.relabel_nodes(G, mapping)

    # Set weights as geodesic distance in meters
    keys = list(coords_dict.keys())
    for a in keys:
        for b in keys:
            if a == b:
                continue
            d_m = geodesic(coords_dict[a], coords_dict[b]).meters
            G.add_edge(a, b, weight=d_m)

    # Solve TSP approximate route starting at School and returning to School
    tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight', nodes=keys, cycle=True)
    # Rotate so 'School' is first
    if tsp_route[0] != "School":
        idx = tsp_route.index("School")
        tsp_route = tsp_route[idx:] + tsp_route[:idx]
    # Ensure route ends back at School
    if tsp_route[-1] != "School":
        tsp_route.append("School")

    # Build pretty tables
    route_rows = []
    for node in tsp_route:
        if node == "School":
            continue
        stop_id = int(node)
        addr = geo_stops.loc[geo_stops["stop_id"] == stop_id, "stop_address"].iloc[0]
        students = df.loc[df["stop_id"] == stop_id, "name"].tolist()
        route_rows.append({"Stop Order": len(route_rows)+1, "Stop ID": stop_id, "Stop Address": addr, "Students": ", ".join(students)})

    st.subheader("Optimized Stop Order")
    st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

    # Step-by-step legs
    legs = []
    cum_m = 0.0
    for i in range(len(tsp_route)-1):
        a, b = tsp_route[i], tsp_route[i+1]
        a_coords = coords_dict[a]
        b_coords = coords_dict[b]

        # Base distance in meters (geodesic)
        base_m = geodesic(a_coords, b_coords).meters
        directions = ""
        seg_m = base_m
        seg_s = None

        # Optional OSRM turn-by-turn
        if use_turn_by_turn:
            txt, dist_m, dur_s = osrm_turn_by_turn(a_coords[0], a_coords[1], b_coords[0], b_coords[1])
            if dist_m is not None and dist_m > 0:
                directions = txt
                seg_m = dist_m
                seg_s = dur_s

        cum_m += seg_m
        # ETA assuming constant speed
        mph = max(average_speed_mph, 1)
        hours = (cum_m / 1609.344) / mph
        eta_minutes = int(round(hours * 60))

        legs.append({
            "From": "School" if a=="School" else f"Stop {a}",
            "To": "School" if b=="School" else f"Stop {b}",
            f"Segment Distance ({'mi' if distance_units=='miles' else 'km'})": round(miles_km(seg_m, distance_units), 2),
            f"Cumulative Distance ({'mi' if distance_units=='miles' else 'km'})": round(miles_km(cum_m, distance_units), 2),
            "ETA (minutes from start)": eta_minutes,
            "Directions": directions
        })

    st.subheader("Driver Sheet (legs)")
    st.dataframe(pd.DataFrame(legs), use_container_width=True, hide_index=True)

    # Map
    st.subheader("Route Map")
    m = folium.Map(location=school_coords, zoom_start=12, control_scale=True)
    folium.Marker(school_coords, popup="School", icon=folium.Icon(color="green")).add_to(m)

    # Stop markers with order labels
    order = 1
    for node in tsp_route:
        if node == "School":
            continue
        coord = coords_dict[node]
        addr = geo_stops.loc[geo_stops["stop_id"] == int(node), "stop_address"].iloc[0]
        students = df.loc[df["stop_id"] == int(node), "name"].tolist()
        folium.Marker(
            location=coord,
            popup=f"#{order} • Stop {node}<br>{addr}<br>{', '.join(students)}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
        order += 1

    # Polyline
    path_coords = [coords_dict[n] for n in tsp_route]
    folium.PolyLine(path_coords, weight=5, opacity=0.8).add_to(m)

    st_folium(m, width=900, height=520)

    # Export: PDF & Excel
    def build_pdf(route_rows, legs):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Dawson's Bus Tool — Driver Sheet", ln=1, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Optimized Stop Order", ln=1)
        for row in route_rows:
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 7, f"#{row['Stop Order']} — Stop {row['Stop ID']}: {row['Stop Address']}")
            pdf.set_font("Arial", "", 11)
            students = row['Students'] or "(no names listed)"
            pdf.multi_cell(0, 6, f"Students: {students}")
            pdf.ln(1)

        pdf.ln(2)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Step-by-step Legs", ln=1)
        for step in legs:
            line = f"{step['From']} → {step['To']} • {step[[k for k in step.keys() if k.startswith('Segment Distance')][0]]} • ETA +{step['ETA (minutes from start)']} min"
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, line)
            if step.get("Directions"):
                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 5, step["Directions"])
            pdf.ln(1)

        return pdf.output(dest="S").encode("latin1")

    pdf_bytes = build_pdf(route_rows, legs)
    st.download_button("⬇️ Download PDF Driver Sheet", data=pdf_bytes, file_name="driver_sheet.pdf", mime="application/pdf")

    # Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(route_rows).to_excel(writer, sheet_name="Stop Order", index=False)
        pd.DataFrame(legs).to_excel(writer, sheet_name="Driver Legs", index=False)
    st.download_button("⬇️ Download Excel Workbook", data=output.getvalue(),
                       file_name="bus_route.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Enter a school address and upload your CSV to begin.")
    st.caption("Tip: Include city and state in addresses for best geocoding results.")
