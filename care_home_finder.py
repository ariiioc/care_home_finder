"""
Care home finder: searches, filters and groups care homes automatically.

You set the boundaries in the sidebar; the app does the rest:
  1. Pulls every registered care home in the boroughs you pick (CQC API)
  2. Keeps those with enough beds and the care type you want
  3. Checks real public-transport time from your start point (TfL)
  4. Groups nearby homes into clusters you can reach on foot or by one bus
  5. Shows them on a map and gives you a spreadsheet to download

Setup (once):
  pip install streamlit requests pandas
  Free CQC key:  https://api-portal.service.cqc.org.uk  (subscribe to "Syndication API")
  Free TfL key:  https://api-portal.tfl.gov.uk
Run:
  streamlit run care_home_finder.py
Optional: set CQC_API_KEY and TFL_API_KEY as environment variables so you don't retype them.
"""

import datetime as dt
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

CQC_BASE = "https://api.service.cqc.org.uk/public/v1"
TFL_BASE = "https://api.tfl.gov.uk"
PLAISTOW = (51.5313, 0.0172)  # Plaistow station

BOROUGHS = [
    "Newham", "Tower Hamlets", "Hackney", "Waltham Forest", "Redbridge",
    "Barking and Dagenham", "Havering", "City of London", "Greenwich",
    "Lewisham", "Islington", "Haringey", "Bexley", "Enfield", "Southwark",
]
DEFAULT_BOROUGHS = BOROUGHS[:8]

GROUP_COLOURS = [
    [31, 119, 180], [214, 39, 40], [44, 160, 44], [148, 103, 189], [255, 127, 14],
    [23, 190, 207], [227, 119, 194], [140, 86, 75], [188, 189, 34], [127, 127, 127],
]


# ---------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_json(url, params=None, headers=None, tries=4):
    """GET with retry on rate limits / transient errors."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (2 ** attempt))
                continue
            if r.status_code in (401, 403):
                raise PermissionError(f"Key rejected by {url.split('/')[2]} (HTTP {r.status_code})")
            if r.status_code != 200:
                return None
            return r.json()
        except requests.RequestException:
            time.sleep(1.5 * (2 ** attempt))
    return None


@st.cache_resource
def store():
    """Results kept between runs, so changing a boundary doesn't re-download everything."""
    return {"cqc": {}, "travel": {}, "bus": {}}


# ---------------------------------------------------------------- CQC
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def cqc_ids_for_borough(cqc_key, borough):
    ids, page = [], 1
    while True:
        d = get_json(
            f"{CQC_BASE}/locations",
            params={"careHome": "Y", "localAuthority": borough, "page": page, "perPage": 1000},
            headers={"Ocp-Apim-Subscription-Key": cqc_key},
        )
        if not d:
            break
        ids += [l["locationId"] for l in d.get("locations", [])]
        if page >= d.get("totalPages", 1):
            break
        page += 1
    return ids


def cqc_detail(cqc_key, loc_id):
    d = get_json(f"{CQC_BASE}/locations/{loc_id}", headers={"Ocp-Apim-Subscription-Key": cqc_key})
    if not d or d.get("registrationStatus") != "Registered":
        return None
    lat, lon = d.get("onspdLatitude"), d.get("onspdLongitude")
    if lat is None or lon is None:
        return None
    services = [s.get("name", "") for s in d.get("gacServiceTypes", [])]
    nursing = any("with nursing" in s.lower() for s in services)
    rating = (d.get("currentRatings") or {}).get("overall", {}).get("rating", "Not rated")
    address = ", ".join(
        x for x in [d.get("postalAddressLine1"), d.get("postalAddressTownCity"), d.get("postalCode")] if x
    )
    return {
        "id": loc_id,
        "name": d.get("name", ""),
        "beds": d.get("numberOfBeds") or 0,
        "nursing": nursing,
        "rating": rating,
        "address": address,
        "postcode": d.get("postalCode", ""),
        "phone": d.get("mainPhoneNumber", ""),
        "specialisms": "; ".join(s.get("name", "") for s in d.get("specialisms", [])),
        "borough": d.get("localAuthority", ""),
        "lat": float(lat),
        "lon": float(lon),
        "cqc_link": f"https://www.cqc.org.uk/location/{loc_id}",
    }


# ---------------------------------------------------------------- TfL
def tfl_minutes(tfl_key, origin, dest, date_str, time_str):
    d = get_json(
        f"{TFL_BASE}/Journey/JourneyResults/{origin[0]},{origin[1]}/to/{dest[0]},{dest[1]}",
        params={"app_key": tfl_key, "date": date_str, "time": time_str, "timeIs": "Departing"},
    )
    if not d or not d.get("journeys"):
        return None
    return min(j["duration"] for j in d["journeys"])


def tfl_bus_lines(tfl_key, lat, lon, radius_m):
    d = get_json(
        f"{TFL_BASE}/StopPoint",
        params={"lat": lat, "lon": lon, "stopTypes": "NaptanPublicBusCoachTram",
                "radius": radius_m, "modes": "bus", "app_key": tfl_key},
    )
    lines = set()
    for sp in (d or {}).get("stopPoints", []):
        for ln in sp.get("lines", []):
            lines.add(ln.get("name", ""))
    lines.discard("")
    return lines


def geocode_postcode(pc):
    d = get_json(f"https://api.postcodes.io/postcodes/{pc.replace(' ', '')}")
    if d and d.get("result"):
        return d["result"]["latitude"], d["result"]["longitude"]
    return None


def run_parallel(fn, items, label, workers=6):
    """Run fn over items with a progress bar; returns {item_key: result}."""
    out, bar = {}, st.progress(0.0, text=label)
    if not items:
        bar.empty()
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, item): key for key, item in items}
        for i, f in enumerate(as_completed(futures), 1):
            out[futures[f]] = f.result()
            bar.progress(i / len(futures), text=f"{label} ({i}/{len(futures)})")
    bar.empty()
    return out


# ---------------------------------------------------------------- grouping
def build_groups(homes, walk_km, allow_bus, min_size, max_size):
    """
    Two homes are 'linked' if they're within walking distance, or (if allowed)
    share a bus route. A group is a set where every home links to every other,
    so you can get between any two on foot or with one bus.
    """
    n = len(homes)
    detour = 1.3  # streets are longer than straight lines
    dist = [[0.0] * n for _ in range(n)]
    link = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(homes[i]["lat"], homes[i]["lon"], homes[j]["lat"], homes[j]["lon"]) * detour
            dist[i][j] = dist[j][i] = d
            if d <= walk_km:
                link[i][j] = link[j][i] = "walk"
            elif allow_bus and homes[i]["bus_lines"] & homes[j]["bus_lines"]:
                link[i][j] = link[j][i] = "bus"

    def cost(i, j):
        return dist[i][j] if link[i][j] == "walk" else dist[i][j] * 2 + 1  # walking preferred

    candidates = []
    for seed in range(n):
        group = [seed]
        pool = {j for j in range(n) if link[seed][j]}
        while len(group) < max_size:
            ok = [c for c in pool if all(link[c][g] for g in group)]
            if not ok:
                break
            best = min(ok, key=lambda c: sum(cost(c, g) for g in group))
            group.append(best)
            pool.discard(best)
        if len(group) >= min_size:
            pairs = [(a, b) for k, a in enumerate(group) for b in group[k + 1:]]
            score = sum(cost(a, b) for a, b in pairs) / len(pairs) - 0.3 * len(group)
            candidates.append((score, frozenset(group)))

    chosen, used, seen = [], set(), set()
    for score, g in sorted(candidates, key=lambda x: x[0]):
        if g in seen or g & used:
            continue
        seen.add(g)
        chosen.append(sorted(g))
        used |= g

    described = []
    for g in chosen:
        common = set.intersection(*(homes[i]["bus_lines"] for i in g)) if g else set()
        walk_pairs = sum(1 for k, a in enumerate(g) for b in g[k + 1:] if link[a][b] == "walk")
        total_pairs = len(g) * (len(g) - 1) // 2
        max_gap = max(dist[a][b] for k, a in enumerate(g) for b in g[k + 1:])
        described.append({
            "members": g,
            "all_walkable": walk_pairs == total_pairs,
            "common_buses": sorted(common),
            "max_gap_km": round(max_gap, 2),
        })
    return described, used


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Care home finder", page_icon="🏠", layout="wide")
st.title("Care home finder")
st.caption("Set your boundaries on the left, then press Find and group care homes.")

with st.sidebar:
    st.header("Access keys")
    cqc_key = st.text_input("CQC key", value=os.getenv("CQC_API_KEY", ""), type="password")
    tfl_key = st.text_input("TfL key", value=os.getenv("TFL_API_KEY", ""), type="password")

    st.header("Where")
    start = st.text_input("Start point", value="Plaistow station",
                          help="Leave as Plaistow station, or type any London postcode")
    boroughs = st.multiselect("Boroughs to search", BOROUGHS, default=DEFAULT_BOROUGHS)
    max_minutes = st.slider("Max travel time from start (minutes)", 15, 120, 60, 5)
    dep_time = st.time_input("Travel time measured leaving at", dt.time(9, 0))

    st.header("Which homes")
    min_beds = st.number_input("Minimum beds", 1, 300, 20)
    care_type = st.radio("Care type", ["Any", "Nursing only", "Residential only (no nursing)"])
    min_rating = st.selectbox("Lowest CQC rating to include",
                              ["Any (incl. not rated)", "Requires improvement", "Good", "Outstanding"])

    st.header("Grouping")
    group_min, group_max = st.slider("Homes per group", 2, 8, (4, 5))
    walk_km = st.slider("Walking distance between homes (km)", 0.3, 2.5, 1.0, 0.1,
                        help="About 1 km is a 12–15 minute walk")
    allow_bus = st.checkbox("Also allow homes linked by a single bus", value=True)
    stop_radius = st.slider("Max walk to a bus stop (metres)", 150, 800, 400, 50)

    go = st.button("Find and group care homes", type="primary", use_container_width=True)

if not go:
    st.info("Add your CQC and TfL keys, adjust the boundaries, then press Find and group care homes.")
    st.stop()

if not cqc_key or not tfl_key:
    st.error("Both keys are needed. Get them free from the CQC and TfL API portals (links at the top of this file).")
    st.stop()
if not boroughs:
    st.error("Pick at least one borough to search.")
    st.stop()

# Start point
if start.strip().lower() in ("plaistow", "plaistow station", ""):
    origin = PLAISTOW
else:
    origin = geocode_postcode(start.strip())
    if not origin:
        st.error(f"Couldn't find the postcode '{start}'. Check it and try again.")
        st.stop()

cache = store()

try:
    # 1. Find every care home in the chosen boroughs
    with st.spinner("Finding registered care homes"):
        ids = []
        for b in boroughs:
            ids += cqc_ids_for_borough(cqc_key, b)
    ids = list(dict.fromkeys(ids))
    if not ids:
        st.error("CQC returned no care homes. Check your CQC key is subscribed to the Syndication API.")
        st.stop()

    todo = [(i, i) for i in ids if i not in cache["cqc"]]
    cache["cqc"].update(run_parallel(lambda i: cqc_detail(cqc_key, i), todo, "Reading care home details", workers=5))
    homes = [cache["cqc"][i] for i in ids if cache["cqc"].get(i)]
except PermissionError as e:
    st.error(f"{e}. Check the key and that it's subscribed to the right product.")
    st.stop()

# 2. Apply your filters
rating_order = {"Outstanding": 4, "Good": 3, "Requires improvement": 2, "Inadequate": 1}
rating_floor = {"Any (incl. not rated)": 0, "Requires improvement": 2, "Good": 3, "Outstanding": 4}[min_rating]
homes = [h for h in homes if h["beds"] >= min_beds]
if care_type == "Nursing only":
    homes = [h for h in homes if h["nursing"]]
elif care_type.startswith("Residential"):
    homes = [h for h in homes if not h["nursing"]]
if rating_floor:
    homes = [h for h in homes if rating_order.get(h["rating"], 0) >= rating_floor]
# Rough distance cut before asking TfL (nothing 25 km+ away is reachable in time)
homes = [h for h in homes if haversine_km(origin[0], origin[1], h["lat"], h["lon"]) <= max_minutes * 0.45]

# 3. Real travel time from the start point
next_weekday = dt.date.today() + dt.timedelta(days=1)
while next_weekday.weekday() >= 5:
    next_weekday += dt.timedelta(days=1)
date_str, time_str = next_weekday.strftime("%Y%m%d"), dep_time.strftime("%H%M")

def travel_key(h):
    return (round(origin[0], 4), round(origin[1], 4), h["id"], time_str)

todo = [(travel_key(h), h) for h in homes if travel_key(h) not in cache["travel"]]
cache["travel"].update(run_parallel(
    lambda h: tfl_minutes(tfl_key, origin, (h["lat"], h["lon"]), date_str, time_str),
    todo, "Checking travel times with TfL"))
for h in homes:
    h["minutes"] = cache["travel"].get(travel_key(h))
homes = [h for h in homes if h["minutes"] is not None and h["minutes"] <= max_minutes]

if not homes:
    st.warning("No care homes match these boundaries. Try more boroughs, fewer beds or a longer travel time.")
    st.stop()

# 4. Bus routes near each home
if allow_bus:
    todo = [((h["id"], stop_radius), h) for h in homes if (h["id"], stop_radius) not in cache["bus"]]
    cache["bus"].update(run_parallel(
        lambda h: tfl_bus_lines(tfl_key, h["lat"], h["lon"], stop_radius), todo, "Finding nearby bus routes"))
for h in homes:
    h["bus_lines"] = cache["bus"].get((h["id"], stop_radius), set()) if allow_bus else set()

# 5. Group them
groups, grouped = build_groups(homes, walk_km, allow_bus, group_min, group_max)

# ---------------------------------------------------------------- results
c1, c2, c3 = st.columns(3)
c1.metric("Homes matching your boundaries", len(homes))
c2.metric("Groups found", len(groups))
c3.metric("Homes placed in a group", len(grouped))

rows = []
for gi, g in enumerate(groups, 1):
    for i in g["members"]:
        h = homes[i]
        rows.append({
            "Group": gi, "Name": h["name"], "Beds": h["beds"],
            "Type": "Nursing" if h["nursing"] else "Residential",
            "CQC rating": h["rating"], "Minutes from start": h["minutes"],
            "Address": h["address"], "Phone": h["phone"], "Borough": h["borough"],
            "Specialisms": h["specialisms"],
            "Group connection": "All walkable" if g["all_walkable"] else (
                "Shared bus: " + ", ".join(g["common_buses"]) if g["common_buses"]
                else "Walk or one bus between each pair"),
            "CQC page": h["cqc_link"], "lat": h["lat"], "lon": h["lon"],
        })
for i, h in enumerate(homes):
    if i not in grouped:
        rows.append({
            "Group": "Not grouped", "Name": h["name"], "Beds": h["beds"],
            "Type": "Nursing" if h["nursing"] else "Residential",
            "CQC rating": h["rating"], "Minutes from start": h["minutes"],
            "Address": h["address"], "Phone": h["phone"], "Borough": h["borough"],
            "Specialisms": h["specialisms"], "Group connection": "",
            "CQC page": h["cqc_link"], "lat": h["lat"], "lon": h["lon"],
        })
df = pd.DataFrame(rows)

# Map
map_df = df.copy()
map_df["color"] = map_df["Group"].apply(
    lambda g: GROUP_COLOURS[(g - 1) % len(GROUP_COLOURS)] if isinstance(g, int) else [200, 200, 200])
map_df["radius"] = map_df["Group"].apply(lambda g: 90 if isinstance(g, int) else 50)
map_df["label"] = map_df.apply(lambda r: f"Group {r['Group']}: {r['Name']} ({r['Beds']} beds)", axis=1)
origin_df = pd.DataFrame([{"lat": origin[0], "lon": origin[1], "label": "Start point"}])

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=origin[0], longitude=origin[1], zoom=11),
    layers=[
        pdk.Layer("ScatterplotLayer", map_df, get_position="[lon, lat]", get_fill_color="color",
                  get_radius="radius", pickable=True, opacity=0.85),
        pdk.Layer("ScatterplotLayer", origin_df, get_position="[lon, lat]",
                  get_fill_color=[0, 0, 0], get_radius=140, pickable=True),
    ],
    tooltip={"text": "{label}"},
))

# Group cards
for gi, g in enumerate(groups, 1):
    sub = df[df["Group"] == gi]
    how = ("all within walking distance" if g["all_walkable"]
           else f"shared bus route(s): {', '.join(g['common_buses'])}" if g["common_buses"]
           else "each pair linked on foot or by one bus")
    with st.expander(f"Group {gi}: {len(sub)} homes, {int(sub['Beds'].sum())} beds total, {how}", expanded=gi <= 3):
        st.dataframe(sub.drop(columns=["Group", "lat", "lon"]), hide_index=True, use_container_width=True)
        st.caption(f"Furthest apart: about {g['max_gap_km']} km by street.")

ungrouped = df[df["Group"] == "Not grouped"]
if len(ungrouped):
    with st.expander(f"{len(ungrouped)} homes match but didn't fit a group"):
        st.dataframe(ungrouped.drop(columns=["Group", "lat", "lon"]), hide_index=True, use_container_width=True)

st.download_button("Download results (CSV)", df.drop(columns=["lat", "lon"]).to_csv(index=False),
                   file_name=f"care_home_groups_{dt.date.today()}.csv", mime="text/csv")
st.caption("Contains CQC data under the Open Government Licence. Travel times from TfL, powered by TfL Open Data.")
