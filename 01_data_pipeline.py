"""
01_data_pipeline.py — Eagles Slot WR Analysis
===============================================
Data acquisition and cleaning pipeline. Run once to build all CSV files
needed for the analysis.

Sources:
  - nflverse (play-by-play, participation, FTN charting, player stats,
    draft picks, combine, rosters, snap counts)
  - MockDraftable (extended measurables: hand size, arm length, wingspan)

Output: data/*.csv
Runtime: ~10-15 minutes (mostly MockDraftable scraping with 1s delay)
"""

import nflreadpy as nfl
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
SEASONS = list(range(2016, 2026))
FTN_SEASONS = list(range(2022, 2026))
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SHANAHAN_TREE = {
    'SF':  list(range(2017, 2026)),
    'LA':  list(range(2017, 2026)),
    'GB':  list(range(2019, 2026)),
    'MIN': list(range(2023, 2026)),
    'DET': list(range(2022, 2026)),
    'MIA': list(range(2022, 2025)),
    'IND': list(range(2023, 2026)),
    'PHI': [2026],
}

def is_shanahan_tree(team, season):
    return season in SHANAHAN_TREE.get(team, [])


# ═══════════════════════════════════════════════════════════
# PART 1: nflverse data
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1: Loading nflverse data")
print("=" * 60)

print("Loading play-by-play...")
pbp = nfl.load_pbp(SEASONS).to_pandas()
print(f"  PBP: {pbp.shape[0]:,} plays, {pbp.shape[1]} columns")

print("Loading participation...")
participation = nfl.load_participation(SEASONS).to_pandas()
print(f"  Participation: {participation.shape[0]:,} rows")

print("Loading FTN charting (2022+)...")
ftn = nfl.load_ftn_charting(FTN_SEASONS).to_pandas()
print(f"  FTN: {ftn.shape[0]:,} rows")

print("Loading player stats...")
player_stats_weekly = nfl.load_player_stats(SEASONS).to_pandas()
player_stats_seasonal = nfl.load_player_stats(SEASONS, summary_level='reg').to_pandas()
print(f"  Weekly: {player_stats_weekly.shape[0]:,} | Seasonal: {player_stats_seasonal.shape[0]:,}")

print("Loading draft picks, combine, rosters, snap counts...")
draft_picks = nfl.load_draft_picks().to_pandas()
combine = nfl.load_combine().to_pandas()
rosters = nfl.load_rosters(SEASONS).to_pandas()
snap_counts = nfl.load_snap_counts(SEASONS).to_pandas()
print(f"  Draft: {len(draft_picks):,} | Combine: {len(combine):,} | Rosters: {len(rosters):,} | Snaps: {len(snap_counts):,}")


# ═══════════════════════════════════════════════════════════
# PART 2: Merge PBP + Participation + FTN
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Merging datasets")
print("=" * 60)

if 'nflverse_game_id' in participation.columns:
    participation = participation.rename(columns={'nflverse_game_id': 'game_id'})

part_cols = ['game_id', 'play_id', 'offense_formation', 'offense_personnel',
             'defenders_in_box', 'defense_personnel', 'number_of_pass_rushers']
part_cols = [c for c in part_cols if c in participation.columns]
pbp_merged = pbp.merge(participation[part_cols].copy(), on=['game_id', 'play_id'], how='left')

if 'nflverse_game_id' in ftn.columns:
    ftn = ftn.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'})
ftn_cols = ['game_id', 'play_id']
for c in ['is_motion', 'is_play_action', 'is_screen_pass', 'is_rpo',
           'is_no_huddle', 'qb_location', 'time_to_throw', 'was_pressure',
           'defense_man_zone_type', 'defense_coverage_type']:
    if c in ftn.columns:
        ftn_cols.append(c)
pbp_merged = pbp_merged.merge(ftn[ftn_cols].copy(), on=['game_id', 'play_id'], how='left')
print(f"  Merged PBP: {pbp_merged.shape}")


# ═══════════════════════════════════════════════════════════
# PART 3: Feature engineering
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Feature engineering")
print("=" * 60)

pbp_merged['is_shanahan_tree'] = pbp_merged.apply(
    lambda r: is_shanahan_tree(r.get('posteam', ''), r.get('season', 0)), axis=1)

def parse_personnel(s):
    if pd.isna(s): return None
    try:
        parts = str(s).split(',')
        rb = te = 0
        for p in parts:
            p = p.strip()
            if 'RB' in p: rb = int(p.split()[0])
            elif 'TE' in p: te = int(p.split()[0])
        return f"{rb}{te}"
    except: return None

pbp_merged['personnel_code'] = pbp_merged['offense_personnel'].apply(parse_personnel)

def tag_yac(row):
    if row.get('is_screen_pass') == True: return 'screen'
    if row.get('is_rpo') == True: return 'rpo'
    desc = str(row.get('desc', '')).lower()
    if any(kw in desc for kw in ['screen', 'bubble', 'jet sweep', 'end around', 'shovel', 'swing pass', 'flare']):
        return 'manufactured_desc'
    air, yac = row.get('air_yards'), row.get('yards_after_catch')
    if air is not None and yac is not None:
        try:
            if float(air) <= 0 and float(yac) > 0: return 'behind_los_yac'
        except: pass
    return 'standard'

pbp_merged['yac_type'] = pbp_merged.apply(tag_yac, axis=1)

# Filter to pass plays with a receiver
pass_plays = pbp_merged[
    (pbp_merged['pass'] == 1) & (pbp_merged['sack'] != 1) & (pbp_merged['receiver'].notna())
].copy()
print(f"  Pass plays: {len(pass_plays):,}")


# ═══════════════════════════════════════════════════════════
# PART 4: Team-season and receiver-season aggregates
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4: Aggregates")
print("=" * 60)

team_season = pbp_merged[
    (pbp_merged['pass'] == 1) | (pbp_merged['rush'] == 1)
].groupby(['posteam', 'season']).agg(
    total_plays=('play_id', 'count'), pass_plays=('pass', 'sum'),
    rush_plays=('rush', 'sum'), mean_epa=('epa', 'mean'),
    shotgun_plays=('shotgun', 'sum'),
    motion_plays=('is_motion', lambda x: x.sum() if x.notna().any() else None),
).reset_index()

team_season['pass_rate'] = team_season['pass_plays'] / team_season['total_plays']
team_season['shotgun_rate'] = team_season['shotgun_plays'] / team_season['total_plays']
team_season['under_center_rate'] = 1 - team_season['shotgun_rate']
team_season['motion_rate'] = team_season['motion_plays'] / team_season['total_plays']
team_season['is_shanahan_tree'] = team_season.apply(
    lambda r: is_shanahan_tree(r['posteam'], r['season']), axis=1)

pers_rates = pbp_merged[pbp_merged['personnel_code'].notna()].groupby(
    ['posteam', 'season', 'personnel_code']).size().reset_index(name='count')
pers_totals = pers_rates.groupby(['posteam', 'season'])['count'].sum().reset_index(name='total')
pers_rates = pers_rates.merge(pers_totals, on=['posteam', 'season'])
pers_rates['rate'] = pers_rates['count'] / pers_rates['total']
twelve = pers_rates[pers_rates['personnel_code'] == '12'][['posteam', 'season', 'rate']].rename(
    columns={'rate': 'twelve_personnel_rate'})
team_season = team_season.merge(twelve, on=['posteam', 'season'], how='left')
team_season['twelve_personnel_rate'] = team_season['twelve_personnel_rate'].fillna(0)

receiver_season = pass_plays.groupby(['receiver_id', 'receiver', 'posteam', 'season']).agg(
    targets=('play_id', 'count'), receptions=('complete_pass', 'sum'),
    yards=('yards_gained', 'sum'), air_yards_total=('air_yards', 'sum'),
    yac_total=('yards_after_catch', 'sum'), mean_epa=('epa', 'mean'),
    total_epa=('epa', 'sum'), mean_cpoe=('cpoe', 'mean'),
    td=('pass_touchdown', 'sum'),
    screen_targets=('yac_type', lambda x: (x != 'standard').sum()),
    motion_targets=('is_motion', lambda x: x.sum() if x.notna().any() else None),
).reset_index()

receiver_season['catch_rate'] = receiver_season['receptions'] / receiver_season['targets']
receiver_season['yac_per_rec'] = receiver_season['yac_total'] / receiver_season['receptions'].replace(0, 1)
receiver_season['adot'] = receiver_season['air_yards_total'] / receiver_season['targets']
receiver_season['screen_target_rate'] = receiver_season['screen_targets'] / receiver_season['targets']
receiver_season['is_shanahan_tree'] = receiver_season.apply(
    lambda r: is_shanahan_tree(r['posteam'], r['season']), axis=1)
receiver_season = receiver_season.merge(
    team_season[['posteam', 'season', 'pass_rate', 'motion_rate', 'twelve_personnel_rate', 'under_center_rate']],
    on=['posteam', 'season'], how='left')

print(f"  Team-seasons: {len(team_season)}")
print(f"  Receiver-seasons: {len(receiver_season):,}")


# ═══════════════════════════════════════════════════════════
# PART 5: MockDraftable extended measurables
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5: Scraping MockDraftable measurables")
print("=" * 60)

FRAC_MAP = {'⅛': 0.125, '¼': 0.25, '⅜': 0.375, '½': 0.5,
            '⅝': 0.625, '¾': 0.75, '⅞': 0.875}

def parse_measurement(raw):
    if not raw: return None
    text = raw.strip()
    if text.endswith('s') or 'reps' in text: return None
    if 'lbs' in text:
        m = re.search(r'([\d.]+)', text)
        return float(m.group(1)) if m else None
    text = text.replace('"', '').replace('\u0027', "'").replace('&quot;', '')
    if "'" in text:
        parts = text.split("'")
        feet = int(parts[0].strip())
        inch_part = parts[1].strip() if len(parts) > 1 else '0'
        digits, frac = '', 0
        for c in inch_part:
            if c.isdigit(): digits += c
            elif c in FRAC_MAP: frac = FRAC_MAP[c]
        return feet * 12 + (int(digits) if digits else 0) + frac
    digits, frac = '', 0
    for c in text:
        if c.isdigit() or c == '.': digits += c
        elif c in FRAC_MAP: frac = FRAC_MAP[c]
    return float(digits) + frac if digits else None

def scrape_player(slug):
    try:
        resp = requests.get(f"https://www.mockdraftable.com/embed/{slug}?position=WR&page=MEASURABLES", timeout=10)
        if resp.status_code != 200: return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        tbody = soup.find('tbody')
        if not tbody: return None
        m = {}
        for row in tbody.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                label, val = cells[0].get_text(strip=True), cells[1].get_text(strip=True)
                if label == 'Hand Size': m['hand_size'] = parse_measurement(val)
                elif label == 'Arm Length': m['arm_length'] = parse_measurement(val)
                elif label == 'Wingspan': m['wingspan'] = parse_measurement(val)
                elif label == 'Height': m['md_height'] = parse_measurement(val)
                elif label == 'Weight': m['md_weight'] = parse_measurement(val)
                elif label == '40 Yard Dash':
                    try: m['forty_md'] = float(val.replace('s',''))
                    except: pass
                elif label == '3-Cone Drill':
                    try: m['cone_md'] = float(val.replace('s',''))
                    except: pass
                elif label == '20 Yard Shuttle':
                    try: m['shuttle_md'] = float(val.replace('s',''))
                    except: pass
                elif label == 'Vertical Jump': m['vertical_md'] = parse_measurement(val)
                elif label == 'Broad Jump': m['broad_jump_md'] = parse_measurement(val)
        return m if m else None
    except Exception as e:
        return None

def name_to_slug(name):
    if pd.isna(name): return None
    slug = re.sub(r"['\.\,]", "", name.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    return re.sub(r"-+", "-", slug)

max_year = draft_picks['season'].max()
wr_draft = draft_picks[
    (draft_picks['position'] == 'WR') & (draft_picks['round'].isin([1, 2, 3])) &
    (draft_picks['season'] >= 2016) & (draft_picks['season'] <= max_year)
].copy()
print(f"  WRs to scrape: {len(wr_draft)}")

md_results = []
for i, (_, wr) in enumerate(wr_draft.iterrows()):
    name, slug = wr['pfr_player_name'], name_to_slug(wr['pfr_player_name'])
    if not slug: continue
    print(f"  [{i+1}/{len(wr_draft)}] {name}...", end=" ")
    data = scrape_player(slug)
    if not data or data.get('hand_size') is None:
        for alt in [name_to_slug(re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name, flags=re.IGNORECASE)),
                    name_to_slug(name.replace(' Jr.', '-jr').replace(' Jr', '-jr')),
                    name_to_slug(name.replace(' III', '-iii'))]:
            if alt and alt != slug:
                data2 = scrape_player(alt)
                if data2 and data2.get('hand_size') is not None:
                    data, slug = data2, alt
                    break
    if data and data.get('hand_size') is not None:
        data.update({'pfr_player_name': name, 'slug': slug, 'gsis_id': wr.get('gsis_id'),
                     'season': wr['season'], 'round': wr['round'], 'pick': wr['pick']})
        md_results.append(data)
        print(f"OK (hand: {data.get('hand_size')})")
    else:
        md_results.append({'pfr_player_name': name, 'slug': slug, 'gsis_id': wr.get('gsis_id'),
                           'season': wr['season'], 'round': wr['round'], 'pick': wr['pick'],
                           'hand_size': None, 'arm_length': None, 'wingspan': None})
        print("NOT FOUND")
    time.sleep(1)

measurables_df = pd.DataFrame(md_results)

# Append manually collected data for known missing players
manual_data = [
    {'pfr_player_name': 'Mike Williams', 'season': 2017, 'round': 1, 'pick': 7,
     'hand_size': 9.375, 'arm_length': 33.375, 'wingspan': 79.75, 'md_height': 75.625, 'md_weight': 218, 'forty_md': 4.54},
    {'pfr_player_name': 'Chris Godwin', 'season': 2017, 'round': 3, 'pick': 84,
     'hand_size': 9.125, 'arm_length': 31.625, 'md_height': 73.0, 'md_weight': 209, 'forty_md': 4.42},
    {'pfr_player_name': 'D.J. Moore', 'season': 2018, 'round': 1, 'pick': 24,
     'hand_size': 9.625, 'arm_length': 31.625, 'wingspan': 76.25, 'md_height': 72.0, 'md_weight': 210, 'forty_md': 4.42},
    {'pfr_player_name': 'John Metchie', 'season': 2022, 'round': 2, 'pick': 44,
     'hand_size': 9.25, 'arm_length': 30.625, 'wingspan': 73.375, 'md_height': 71.25, 'md_weight': 187},
    {'pfr_player_name': 'Marvin Mims', 'season': 2023, 'round': 2, 'pick': 63,
     'hand_size': 9.0, 'arm_length': 31.625, 'wingspan': 74.75, 'md_height': 70.875, 'md_weight': 183, 'forty_md': 4.38},
    {'pfr_player_name': 'Tank Dell', 'season': 2023, 'round': 3, 'pick': 69,
     'hand_size': 8.625, 'arm_length': 30.5, 'wingspan': 72.625, 'md_height': 68.375, 'md_weight': 165, 'forty_md': 4.49},
    {'pfr_player_name': 'Malik Nabers', 'season': 2024, 'round': 1, 'pick': 6,
     'hand_size': 9.875, 'arm_length': 31.375, 'wingspan': 76.125, 'md_height': 72.25, 'md_weight': 199, 'forty_md': 4.35},
    {'pfr_player_name': 'Luther Burden', 'season': 2025, 'round': 1, 'pick': 16,
     'hand_size': 8.5, 'arm_length': 31.25, 'md_height': 72.0, 'md_weight': 206, 'forty_md': 4.41},
    {'pfr_player_name': 'Chad Williams', 'season': 2017, 'round': 3, 'pick': 98,
     'hand_size': 9.75, 'arm_length': 32.0, 'md_height': 72.5, 'md_weight': 207, 'forty_md': 4.43},
]

for player in manual_data:
    match = draft_picks[(draft_picks['pfr_player_name'] == player['pfr_player_name']) &
                        (draft_picks['season'] == player['season'])]
    player['gsis_id'] = match.iloc[0]['gsis_id'] if len(match) > 0 else None

names_manual = [p['pfr_player_name'] for p in manual_data]
measurables_df = measurables_df[~measurables_df['pfr_player_name'].isin(names_manual)]
measurables_df = pd.concat([measurables_df, pd.DataFrame(manual_data)], ignore_index=True)

print(f"\n  Measurables: {len(measurables_df)} players, {measurables_df['hand_size'].notna().sum()} with hand size")


# ═══════════════════════════════════════════════════════════
# PART 6: Build WR career profiles (years 1-4)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 6: Building WR career profiles")
print("=" * 60)

wr_draft_profiles = draft_picks[
    (draft_picks['position'] == 'WR') & (draft_picks['round'].isin([1, 2, 3])) &
    (draft_picks['season'] >= 2016) & (draft_picks['season'] <= 2022)
].copy()

# Merge measurables
ext_cols = [c for c in ['pfr_player_name', 'season', 'hand_size', 'arm_length', 'wingspan',
            'md_height', 'md_weight', 'forty_md', 'cone_md', 'shuttle_md',
            'vertical_md', 'broad_jump_md'] if c in measurables_df.columns]
wr_draft_profiles = wr_draft_profiles.merge(
    measurables_df[ext_cols].drop_duplicates(subset=['pfr_player_name', 'season']),
    on=['pfr_player_name', 'season'], how='left')

# Also merge nflverse combine as fallback
def parse_height(s):
    if pd.isna(s): return np.nan
    try:
        p = str(s).split('-')
        return int(p[0]) * 12 + int(p[1])
    except: return np.nan

combine_wr = combine[combine['pos'] == 'WR'].copy()
combine_wr['height_inches'] = combine_wr['ht'].apply(parse_height)
wr_draft_profiles = wr_draft_profiles.merge(
    combine_wr[['pfr_id', 'height_inches', 'wt', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle']],
    left_on='pfr_player_id', right_on='pfr_id', how='left')

# Consolidate: prefer MockDraftable, fallback to nflverse
wr_draft_profiles['height_inches'] = wr_draft_profiles.get('md_height', pd.Series(dtype=float)).fillna(wr_draft_profiles['height_inches'])
wr_draft_profiles['wt'] = wr_draft_profiles.get('md_weight', pd.Series(dtype=float)).fillna(wr_draft_profiles['wt'])
if 'forty_md' in wr_draft_profiles.columns:
    wr_draft_profiles['forty'] = wr_draft_profiles['forty_md'].fillna(wr_draft_profiles['forty'])
if 'vertical_md' in wr_draft_profiles.columns:
    wr_draft_profiles['vertical'] = wr_draft_profiles['vertical_md'].fillna(wr_draft_profiles['vertical'])
if 'broad_jump_md' in wr_draft_profiles.columns:
    wr_draft_profiles['broad_jump'] = wr_draft_profiles['broad_jump_md'].fillna(wr_draft_profiles['broad_jump'])

for col in ['height_inches', 'wt', 'forty', 'vertical', 'broad_jump', 'hand_size', 'arm_length']:
    if col in wr_draft_profiles.columns:
        wr_draft_profiles[col] = wr_draft_profiles[col].fillna(wr_draft_profiles[col].median())

# Build career stats
wr_stats = player_stats_seasonal[
    (player_stats_seasonal['position'] == 'WR') & (player_stats_seasonal['season_type'] == 'REG')
].copy()

career_stats = []
for _, wr in wr_draft_profiles.iterrows():
    gsis_id = wr['gsis_id']
    if pd.isna(gsis_id): continue
    ps = wr_stats[(wr_stats['player_id'] == gsis_id) &
                   (wr_stats['season'] >= wr['season']) &
                   (wr_stats['season'] <= wr['season'] + 3)].sort_values('season')
    if len(ps) == 0: continue
    tg = ps['games'].sum()
    career_stats.append({
        'gsis_id': gsis_id, 'name': wr['pfr_player_name'],
        'draft_season': wr['season'], 'draft_round': wr['round'], 'draft_pick': wr['pick'],
        'height_inches': wr['height_inches'], 'weight': wr['wt'],
        'forty': wr['forty'], 'vertical': wr['vertical'], 'broad_jump': wr['broad_jump'],
        'hand_size': wr.get('hand_size', np.nan), 'arm_length': wr.get('arm_length', np.nan),
        'n_seasons': len(ps), 'total_games': tg,
        'total_targets': ps['targets'].sum(), 'total_receptions': ps['receptions'].sum(),
        'total_yards': ps['receiving_yards'].sum(), 'total_tds': ps['receiving_tds'].sum(),
        'total_yac': ps['receiving_yards_after_catch'].sum(), 'total_epa': ps['receiving_epa'].sum(),
        'targets_per_game': ps['targets'].sum() / max(tg, 1),
        'yards_per_game': ps['receiving_yards'].sum() / max(tg, 1),
        'catch_rate': ps['receptions'].sum() / max(ps['targets'].sum(), 1),
        'yards_per_catch': ps['receiving_yards'].sum() / max(ps['receptions'].sum(), 1),
        'yac_per_catch': ps['receiving_yards_after_catch'].sum() / max(ps['receptions'].sum(), 1),
        'yr1_targets': ps.iloc[0]['targets'], 'yr1_receptions': ps.iloc[0]['receptions'],
        'yr1_yards': ps.iloc[0]['receiving_yards'], 'yr1_tds': ps.iloc[0]['receiving_tds'],
        'yr1_games': ps.iloc[0]['games'], 'yr1_yac': ps.iloc[0]['receiving_yards_after_catch'],
        'yr1_epa': ps.iloc[0]['receiving_epa'],
    })

career_df = pd.DataFrame(career_stats)
print(f"  WR career profiles: {len(career_df)}")


# ═══════════════════════════════════════════════════════════
# PART 7: Save everything
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 7: Saving all datasets")
print("=" * 60)

pass_plays.to_csv(DATA_DIR / "pass_plays_enriched.csv", index=False)
team_season.to_csv(DATA_DIR / "team_season_features.csv", index=False)
receiver_season.to_csv(DATA_DIR / "receiver_season_features.csv", index=False)
measurables_df.to_csv(DATA_DIR / "wr_extended_measurables.csv", index=False)
career_df.to_csv(DATA_DIR / "wr_career_profiles.csv", index=False)
draft_picks.to_csv(DATA_DIR / "draft_picks.csv", index=False)
combine.to_csv(DATA_DIR / "combine.csv", index=False)
player_stats_seasonal.to_csv(DATA_DIR / "player_stats_seasonal.csv", index=False)
rosters.to_csv(DATA_DIR / "rosters.csv", index=False)
snap_counts.to_csv(DATA_DIR / "snap_counts.csv", index=False)

print(f"  Saved {len(list(DATA_DIR.glob('*.csv')))} CSV files to {DATA_DIR}/")
for f in sorted(DATA_DIR.glob('*.csv')):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"    {f.name}: {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("DATA PIPELINE COMPLETE")
print("Run 02_analysis.py next.")
print("=" * 60)