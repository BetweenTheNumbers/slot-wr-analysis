"""
02_analysis.py — Eagles Slot WR Analysis
==========================================
Research question:
  "How does the Shanahan-tree offensive scheme change slot WR usage
  and efficiency, and what does this mean for how the Eagles should
  deploy Makai Lemon in 2026?"

Parts:
  1. Scheme EDA — motion, personnel, formation differences
  2. PyTorch EPA model — play-level efficiency prediction
  3. Player quality model — draft profile → production (Ridge + model selection)
  4. Optimal usage — what play designs maximize slot WR EPA?
  5. Monte Carlo projection — Lemon season simulations with uncertainty

Run after 01_data_pipeline.py has generated data/*.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
N_SIMULATIONS = 2000
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 150,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.dpi': 200,
})

EAGLES_GREEN = '#004C54'
EAGLES_SILVER = '#A5ACAF'
HIGHLIGHT = '#e74c3c'
SHANAHAN_COLOR = '#2ecc71'
OTHER_COLOR = '#95a5a6'
GOLD = '#d4a017'

# Lemon's combine measurables
LEMON = {
    'draft_round': 1, 'draft_pick': 20, 'height_inches': 71.125,
    'weight': 192.0, 'forty': 4.53, 'vertical': 37.0, 'broad_jump': 123.0,
    'hand_size': 8.75, 'arm_length': 30.5, 'wingspan': 73.25,
}


# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Loading processed data...")
print("=" * 60)

pass_plays = pd.read_csv("data/pass_plays_enriched.csv")
team_season = pd.read_csv("data/team_season_features.csv")
receiver_season = pd.read_csv("data/receiver_season_features.csv")
career_df = pd.read_csv("data/wr_career_profiles.csv")

# Fix CSV round-trip types
for df in [team_season, receiver_season, pass_plays]:
    if 'is_shanahan_tree' in df.columns:
        df['is_shanahan_tree'] = df['is_shanahan_tree'].astype(bool)
if 'personnel_code' in pass_plays.columns:
    pass_plays['personnel_code'] = pass_plays['personnel_code'].astype(str).str.replace('.0', '', regex=False)

print(f"  Pass plays: {len(pass_plays):,}")
print(f"  Team-seasons: {len(team_season)}")
print(f"  Receiver-seasons: {len(receiver_season):,}")
print(f"  WR career profiles: {len(career_df)}")


# ╔═══════════════════════════════════════════════════════════╗
# ║  PART 1: SCHEME EDA                                       ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 1: SCHEME EDA — Shanahan-Tree vs. League")
print("=" * 60)

# 1a. Motion rate comparison (2022+)
motion_data = team_season[(team_season['season'] >= 2022) & (team_season['motion_rate'].notna())]
shan_motion = motion_data[motion_data['is_shanahan_tree']]['motion_rate'].mean()
other_motion = motion_data[~motion_data['is_shanahan_tree']]['motion_rate'].mean()
print(f"\n  Motion rate (2022-2025):")
print(f"    Shanahan Tree: {shan_motion:.1%}  |  Rest of NFL: {other_motion:.1%}  |  Diff: +{shan_motion - other_motion:.1%}")

# 1b. Under center rate
uc_shan = team_season[(team_season['is_shanahan_tree']) & (team_season['season'] >= 2022)]['under_center_rate'].mean()
uc_other = team_season[(~team_season['is_shanahan_tree']) & (team_season['season'] >= 2022)]['under_center_rate'].mean()
print(f"  Under center rate (2022-2025):")
print(f"    Shanahan Tree: {uc_shan:.1%}  |  Rest of NFL: {uc_other:.1%}")

# 1c. 12 personnel usage
t12_shan = team_season[(team_season['is_shanahan_tree']) & (team_season['season'] >= 2022)]['twelve_personnel_rate'].mean()
t12_other = team_season[(~team_season['is_shanahan_tree']) & (team_season['season'] >= 2022)]['twelve_personnel_rate'].mean()
print(f"  12 personnel rate (2022-2025):")
print(f"    Shanahan Tree: {t12_shan:.1%}  |  Rest of NFL: {t12_other:.1%}")

# Figure 1: Motion rate by scheme
fig, ax = plt.subplots(figsize=(10, 6))
for season in sorted(motion_data['season'].unique()):
    s = motion_data[motion_data['season'] == season]
    ax.scatter([season - 0.1] * len(s[s['is_shanahan_tree']]), s[s['is_shanahan_tree']]['motion_rate'],
               color=SHANAHAN_COLOR, alpha=0.7, s=50, label='Shanahan Tree' if season == 2022 else None)
    ax.scatter([season + 0.1] * len(s[~s['is_shanahan_tree']]), s[~s['is_shanahan_tree']]['motion_rate'],
               color=OTHER_COLOR, alpha=0.4, s=30, label='Rest of NFL' if season == 2022 else None)
for is_shan, color in [(True, SHANAHAN_COLOR), (False, OTHER_COLOR)]:
    means = motion_data.groupby(['season', 'is_shanahan_tree'])['motion_rate'].mean().reset_index()
    sub = means[means['is_shanahan_tree'] == is_shan]
    ax.plot(sub['season'], sub['motion_rate'], color=color, linewidth=2.5, marker='D', markersize=8, zorder=4)
ax.set_xlabel('Season'); ax.set_ylabel('Pre-Snap Motion Rate')
ax.set_title('Shanahan-Tree Offenses Use Significantly More Pre-Snap Motion')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '01_motion_rate.png'); plt.close()
print("  Saved: 01_motion_rate.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║  PART 2: PYTORCH EPA MODEL                                ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 2: PYTORCH EPA MODEL")
print("=" * 60)

# Prepare features
df = pass_plays.copy()
df['is_shotgun'] = df['shotgun'].fillna(0).astype(float)
df['is_no_huddle'] = df['no_huddle'].fillna(0).astype(float) if 'no_huddle' in df.columns else 0.0
df['is_shanahan'] = df['is_shanahan_tree'].astype(float)
df['is_12_personnel'] = (df['personnel_code'] == '12').astype(float)
df['is_11_personnel'] = (df['personnel_code'] == '11').astype(float)
df['is_13_personnel'] = (df['personnel_code'] == '13').astype(float)
df['has_motion'] = df['is_motion'].fillna(0).astype(float)
df['has_play_action'] = df['is_play_action'].fillna(0).astype(float)
df['is_screen'] = df['is_screen_pass'].fillna(0).astype(float)
df['is_rpo_play'] = df['is_rpo'].fillna(0).astype(float)
df['down'] = df['down'].fillna(0).astype(float)
df['ydstogo'] = df['ydstogo'].fillna(10).astype(float)
df['yardline_100'] = df['yardline_100'].fillna(50).astype(float)
df['half_seconds_remaining'] = df['half_seconds_remaining'].fillna(900).astype(float)
df['score_diff'] = df['score_differential'].fillna(0).astype(float) if 'score_differential' in df.columns else 0.0
df['pass_left'] = (df['pass_location'] == 'left').astype(float)
df['pass_middle'] = (df['pass_location'] == 'middle').astype(float)
df['pass_right'] = (df['pass_location'] == 'right').astype(float)
df['air_yards_clean'] = df['air_yards'].fillna(df['air_yards'].median()).clip(-10, 60).astype(float)
df['box_count'] = df['defenders_in_box'].fillna(6).astype(float)
df = df.dropna(subset=['epa'])

FEATURE_COLS = [
    'is_shotgun', 'is_no_huddle', 'is_shanahan',
    'is_12_personnel', 'is_11_personnel', 'is_13_personnel',
    'has_motion', 'has_play_action', 'is_screen', 'is_rpo_play',
    'down', 'ydstogo', 'yardline_100', 'half_seconds_remaining',
    'score_diff', 'pass_left', 'pass_middle', 'pass_right',
    'air_yards_clean', 'box_count',
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['epa'].values.astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
scaler_epa = StandardScaler()
X_train_s = scaler_epa.fit_transform(X_train)
X_test_s = scaler_epa.transform(X_test)

X_train_t = torch.FloatTensor(X_train_s).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
X_test_t = torch.FloatTensor(X_test_s).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=1024, shuffle=True)

class EPAPredictor(nn.Module):
    """
    Two-headed neural net predicting mean AND variance of EPA.
    Trained with Gaussian negative log-likelihood so the model
    learns both what to expect AND how uncertain it is.
    """
    def __init__(self, n_features, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        self.shared = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dims[-1], 1)
        self.log_var_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.log_var_head(h)

def gaussian_nll(mu, log_var, target):
    var = torch.exp(log_var) + 1e-6
    return torch.mean(0.5 * (log_var + (target - mu) ** 2 / var))

model = EPAPredictor(n_features=len(FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"  Training EPA model ({sum(p.numel() for p in model.parameters()):,} params)...")
best_loss = float('inf')
for epoch in range(60):
    model.train()
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        mu, lv = model(Xb)
        loss = gaussian_nll(mu, lv, yb)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        mu_t, lv_t = model(X_test_t)
        test_loss = gaussian_nll(mu_t, lv_t, y_test_t).item()
        ss_res = ((y_test_t - mu_t) ** 2).sum().item()
        ss_tot = ((y_test_t - y_test_t.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'outputs/best_epa_model.pt')
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}: test_loss={test_loss:.4f}, R²={r2:.4f}")

model.load_state_dict(torch.load('outputs/best_epa_model.pt', weights_only=True))
model.eval()
with torch.no_grad():
    mu_t, _ = model(X_test_t)
    final_r2 = 1 - ((y_test_t - mu_t) ** 2).sum().item() / ((y_test_t - y_test_t.mean()) ** 2).sum().item()
print(f"  Final test R²: {final_r2:.4f}")

# Feature importance (gradient attribution)
X_test_g = torch.FloatTensor(X_test_s).to(DEVICE).requires_grad_(True)
mu_out, _ = model(X_test_g)
mu_out.sum().backward()
importance = X_test_g.grad.abs().mean(dim=0).cpu().numpy()
importance = importance / importance.sum()

fig, ax = plt.subplots(figsize=(10, 7))
idx = np.argsort(importance)
ax.barh(range(len(idx)), importance[idx], color=EAGLES_GREEN, alpha=0.8)
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([FEATURE_COLS[i] for i in idx], fontsize=9)
ax.set_xlabel('Relative Importance')
ax.set_title('PyTorch EPA Model: Feature Importance')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '02_epa_feature_importance.png'); plt.close()
print("  Saved: 02_epa_feature_importance.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║  PART 3: PLAYER QUALITY MODEL                             ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 3: PLAYER QUALITY MODEL — Draft Profile → Production")
print("=" * 60)

model_df = career_df[career_df['yr1_games'] >= 8].copy()
model_df['log_pick'] = np.log(model_df['draft_pick'])
model_df['speed_score'] = model_df['weight'] * (10 / model_df['forty'].clip(4.2, 5.0)) ** 2
model_df['bmi_proxy'] = model_df['weight'] / (model_df['height_inches'] ** 2) * 703
model_df['yr1_ypg'] = model_df['yr1_yards'] / model_df['yr1_games']

PROFILE_FEATURES = [
    'draft_round', 'log_pick', 'height_inches', 'weight',
    'forty', 'vertical', 'broad_jump', 'speed_score', 'bmi_proxy',
    'hand_size', 'arm_length',
]

X_prof = model_df[PROFILE_FEATURES].values.astype(np.float32)
y_ypg = model_df['yr1_ypg'].values.astype(np.float32)

scaler_prof = StandardScaler()
X_prof_s = scaler_prof.fit_transform(X_prof)

# Model comparison
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
print(f"\n  Model comparison ({len(model_df)} WRs, {len(PROFILE_FEATURES)} features):")
print(f"  {'Model':<30s} {'CV R²':>10s} {'In-sample':>10s}")
print(f"  {'-'*55}")

comparison_results = {}
for name, mdl in [('OLS', LinearRegression()),
                   ('Ridge (α=1)', Ridge(alpha=1.0)),
                   ('Lasso (α=0.1)', Lasso(alpha=0.1, max_iter=10000)),
                   ('ElasticNet (α=0.1)', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))]:
    cv_scores = cross_val_score(mdl, X_prof_s, y_ypg, cv=cv, scoring='r2')
    mdl.fit(X_prof_s, y_ypg)
    in_r2 = 1 - np.sum((y_ypg - mdl.predict(X_prof_s)) ** 2) / np.sum((y_ypg - y_ypg.mean()) ** 2)
    comparison_results[name] = {'cv_r2': cv_scores.mean(), 'in_r2': in_r2}
    print(f"  {name:<30s} {cv_scores.mean():+.3f}      {in_r2:.3f}")

# CV-optimal Ridge
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=cv, scoring='r2')
ridge_cv.fit(X_prof_s, y_ypg)
cv_scores = cross_val_score(ridge_cv, X_prof_s, y_ypg, cv=cv, scoring='r2')
in_r2 = 1 - np.sum((y_ypg - ridge_cv.predict(X_prof_s)) ** 2) / np.sum((y_ypg - y_ypg.mean()) ** 2)
print(f"  {'Ridge (CV-optimal, α=' + f'{ridge_cv.alpha_:.1f})':<30s} {cv_scores.mean():+.3f}      {in_r2:.3f}  ← SELECTED")

# Lasso path for feature importance
print(f"\n  Lasso feature survival (robustness ranking):")
alphas = np.logspace(-3, 1.5, 100)
coef_paths = np.array([Lasso(alpha=a, max_iter=10000).fit(X_prof_s, y_ypg).coef_ for a in alphas])
for j, feat in enumerate(PROFILE_FEATURES):
    nonzero = np.where(np.abs(coef_paths[:, j]) > 1e-6)[0]
    last = alphas[np.max(nonzero)] if len(nonzero) > 0 else 0
    print(f"    {feat:<20s}: survives to α={last:.3f}")

# Apply to Lemon
LEMON['log_pick'] = np.log(LEMON['draft_pick'])
LEMON['speed_score'] = LEMON['weight'] * (10 / LEMON['forty']) ** 2
LEMON['bmi_proxy'] = LEMON['weight'] / (LEMON['height_inches'] ** 2) * 703

lemon_x = np.array([[LEMON[f] for f in PROFILE_FEATURES]], dtype=np.float32)
lemon_pred = ridge_cv.predict(scaler_prof.transform(lemon_x))[0]
avg_ypg = model_df['yr1_ypg'].mean()
lemon_mult = np.clip(lemon_pred / avg_ypg, 0.7, 1.5)

print(f"\n  Lemon projection: {lemon_pred:.1f} ypg (avg: {avg_ypg:.1f}, multiplier: {lemon_mult:.2f}x)")

# Closest comps
model_df['dist'] = np.sqrt(
    ((model_df['draft_pick'] - 20) / 50) ** 2 + ((model_df['height_inches'] - 71.125) / 5) ** 2 +
    ((model_df['weight'] - 192) / 20) ** 2 + ((model_df['forty'] - 4.53) / 0.2) ** 2 +
    ((model_df['hand_size'] - 8.75) / 0.5) ** 2 + ((model_df['arm_length'] - 30.5) / 1.5) ** 2
)
comps = model_df.nsmallest(8, 'dist')
print(f"\n  Closest comps:")
print(comps[['name', 'draft_season', 'draft_pick', 'height_inches', 'weight', 'forty',
             'hand_size', 'yr1_yards', 'total_yards']].to_string(index=False))

# Figure: Comps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(model_df['draft_pick'], model_df['yr1_yards'], color=OTHER_COLOR, alpha=0.5, s=30, label='All WRs')
ax1.scatter(comps['draft_pick'], comps['yr1_yards'], color=SHANAHAN_COLOR, s=80, edgecolors='white', linewidth=1.5, label='Lemon comps', zorder=5)
for _, c in comps.iterrows():
    ax1.annotate(c['name'].split(' ')[-1], xy=(c['draft_pick'], c['yr1_yards']), fontsize=7, textcoords='offset points', xytext=(5, 3))
ax1.axvline(x=20, color=EAGLES_GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Draft Pick'); ax1.set_ylabel('Year 1 Yards'); ax1.set_title('Draft Capital vs. Rookie Production'); ax1.legend(fontsize=8)

ax2.scatter(model_df['hand_size'], model_df['yr1_ypg'], color=OTHER_COLOR, alpha=0.5, s=30)
ax2.scatter(comps['hand_size'], comps['yr1_ypg'], color=SHANAHAN_COLOR, s=80, edgecolors='white', linewidth=1.5, zorder=5)
ax2.axvline(x=8.75, color=EAGLES_GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
ax2.annotate('Lemon 8¾"', xy=(8.75, ax2.get_ylim()[1] * 0.9), fontsize=9, color=EAGLES_GREEN, fontweight='bold', ha='center')
ax2.set_xlabel('Hand Size (in)'); ax2.set_ylabel('Year 1 Yards/Game'); ax2.set_title('Hand Size vs. Rookie Efficiency')
plt.suptitle('Makai Lemon: Draft Profile Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '03_lemon_comps.png'); plt.close()
print("  Saved: 03_lemon_comps.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║  PART 4: OPTIMAL USAGE ANALYSIS                            ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 4: OPTIMAL USAGE — What Maximizes Slot WR EPA?")
print("=" * 60)

# Identify slot WR targets via ADOT
slot_rcvrs = receiver_season[(receiver_season['adot'] < 10) & (receiver_season['adot'] > 0) &
                              (receiver_season['targets'] >= 40)][['receiver_id', 'season']].drop_duplicates()
pass_plays['is_slot_target'] = pass_plays.set_index(['receiver_id', 'season']).index.isin(
    slot_rcvrs.set_index(['receiver_id', 'season']).index)

# Alpha WR1 tagging (120+ targets)
alpha_rcvrs = receiver_season[receiver_season['targets'] >= 120].copy()
alpha_teams = pass_plays[pass_plays['receiver_id'].isin(alpha_rcvrs['receiver_id'])][
    ['receiver_id', 'posteam', 'season']].drop_duplicates()
alpha_set = set(zip(alpha_teams['posteam'], alpha_teams['season']))
pass_plays['team_has_alpha_wr'] = [x in alpha_set for x in zip(pass_plays['posteam'], pass_plays['season'])]

slot_targets = pass_plays[(pass_plays['is_slot_target']) & (pass_plays['epa'].notna())].copy()
shan_slot = slot_targets[slot_targets['is_shanahan_tree']].copy()

print(f"  Slot WR targets: {len(slot_targets):,} (Shanahan: {len(shan_slot):,})")

# Play design categories
def categorize_play(row):
    if row.get('is_screen_pass') == True or row.get('yac_type') == 'screen': return 'Screen'
    if row.get('is_rpo') == True or row.get('yac_type') == 'rpo': return 'RPO'
    if row.get('is_play_action') == True: return 'Play Action'
    air = row.get('air_yards', 5)
    if pd.notna(air):
        if air <= 0: return 'Behind LOS'
        elif air <= 5: return 'Quick Game (1-5)'
        elif air <= 10: return 'Short (6-10)'
        elif air <= 20: return 'Intermediate (11-20)'
        else: return 'Deep (20+)'
    return 'Unknown'

shan_slot['play_category'] = shan_slot.apply(categorize_play, axis=1)

# 4a. Play design EPA ranking
play_epa = shan_slot.groupby('play_category').agg(
    epa=('epa', 'mean'), n=('epa', 'count'), comp=('complete_pass', 'mean'),
    yac=('yards_after_catch', 'mean'), yds=('yards_gained', 'mean')).reset_index().sort_values('epa', ascending=False)

print(f"\n  Play design EPA (Shanahan slot WRs):")
print(f"  {'Category':<22s} {'EPA':>7s} {'n':>6s} {'Comp%':>6s} {'Yds':>5s}")
for _, r in play_epa.iterrows():
    print(f"  {r['play_category']:<22s} {r['epa']:+.3f} {r['n']:>5.0f} {r['comp']:.0%} {r['yds']:.1f}")

# 4b. Motion lift
motion_plays = pass_plays[(pass_plays['season'] >= 2022) & (pass_plays['is_motion'].notna()) & (pass_plays['epa'].notna())].copy()
motion_plays['is_slot_target'] = motion_plays.set_index(['receiver_id', 'season']).index.isin(
    slot_rcvrs.set_index(['receiver_id', 'season']).index)

print(f"\n  Motion EPA lift (motion - no motion):")
for scheme, sv in [('Shanahan', True), ('Other', False)]:
    for wr, wv in [('Slot', True), ('Outside', False)]:
        m = motion_plays[(motion_plays['is_shanahan_tree'] == sv) & (motion_plays['is_slot_target'] == wv)]
        mot = m[m['is_motion'] == True]['epa'].mean()
        no_mot = m[m['is_motion'] == False]['epa'].mean()
        if len(m[m['is_motion'] == True]) > 50:
            print(f"    {scheme} / {wr}: {mot - no_mot:+.3f} EPA/play")

# 4c. Alpha WR1 effect
print(f"\n  Alpha WR1 effect on slot WR EPA (Shanahan):")
for has_alpha, label in [(True, 'With alpha'), (False, 'Without alpha')]:
    sub = shan_slot[shan_slot['team_has_alpha_wr'] == has_alpha]
    if len(sub) > 50:
        print(f"    {label:<18s}: {sub['epa'].mean():+.3f} EPA (n={len(sub):,})")

# 4d. Personnel effect
print(f"\n  Personnel effect on slot WR EPA (Shanahan):")
for pers in ['11', '12', '13']:
    sub = shan_slot[shan_slot['personnel_code'] == pers]
    if len(sub) > 50:
        label = {'11': '11 (3 WR)', '12': '12 (2 TE)', '13': '13 (3 TE)'}.get(pers, pers)
        print(f"    {label:<12s}: {sub['epa'].mean():+.3f} EPA (n={len(sub):,})")

# 4e. Pass location
print(f"\n  Pass location EPA (Shanahan slot WRs):")
for loc in ['left', 'middle', 'right']:
    sub = shan_slot[shan_slot['pass_location'] == loc]
    if len(sub) > 50:
        print(f"    {loc:<8s}: {sub['epa'].mean():+.3f} (n={len(sub):,}, comp={sub['complete_pass'].mean():.0%})")

# Figure: Play design + motion lift
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
cats = play_epa[~play_epa['play_category'].isin(['Unknown'])].sort_values('epa', ascending=True)
colors = [EAGLES_GREEN if e > 0 else HIGHLIGHT for e in cats['epa']]
ax1.barh(range(len(cats)), cats['epa'], color=colors, edgecolor='white', height=0.7)
ax1.set_yticks(range(len(cats))); ax1.set_yticklabels(cats['play_category'], fontsize=10)
ax1.set_xlabel('EPA per Target'); ax1.set_title('Slot WR EPA by Play Design\n(Shanahan-Tree)')
ax1.axvline(x=0, color='gray', linewidth=0.5)
for i, (_, r) in enumerate(cats.iterrows()):
    ax1.text(max(r['epa'], 0) + 0.02, i, f"n={r['n']:.0f}", fontsize=8, va='center', color='gray')

# Motion lift panel
lifts, labels_m, colors_m = [], [], []
for scheme, sv in [('Shanahan', True), ('Other', False)]:
    for wr, wv in [('Slot', True), ('Outside', False)]:
        m = motion_plays[(motion_plays['is_shanahan_tree'] == sv) & (motion_plays['is_slot_target'] == wv)]
        mot = m[m['is_motion'] == True]['epa'].mean()
        no_mot = m[m['is_motion'] == False]['epa'].mean()
        if len(m[m['is_motion'] == True]) > 50:
            lifts.append(mot - no_mot)
            labels_m.append(f"{scheme}\n{wr}")
            colors_m.append(SHANAHAN_COLOR if sv else OTHER_COLOR)

ax2.bar(range(len(lifts)), lifts, color=colors_m, edgecolor='white')
ax2.set_xticks(range(len(lifts))); ax2.set_xticklabels(labels_m, fontsize=9)
ax2.set_ylabel('EPA Lift from Motion'); ax2.set_title('Motion is a Slot WR Weapon')
ax2.axhline(y=0, color='gray', linewidth=0.5)
for i, v in enumerate(lifts):
    ax2.text(i, v + 0.003, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('How Should the Eagles Use Makai Lemon?', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '04_optimal_usage.png'); plt.close()
print("  Saved: 04_optimal_usage.png")

# Figure: Alpha WR1 + Personnel + Location
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))

# Alpha effect
for i, (has_alpha, label, color) in enumerate([(True, 'With Alpha\n(Brown)', EAGLES_GREEN),
                                                 (False, 'Without Alpha\n(Brown traded)', HIGHLIGHT)]):
    sub = shan_slot[shan_slot['team_has_alpha_wr'] == has_alpha]
    if len(sub) > 50:
        se = sub['epa'].std() / np.sqrt(len(sub))
        ax1.bar(i, sub['epa'].mean(), color=color, yerr=se*1.96, capsize=5, edgecolor='white', width=0.6)
        ax1.text(i, sub['epa'].mean() + se*2 + 0.02, f"{sub['epa'].mean():+.3f}\nn={len(sub):,}",
                ha='center', fontsize=9, fontweight='bold')
ax1.set_xticks([0, 1]); ax1.set_xticklabels(['With Alpha\n(Brown)', 'Without Alpha\n(Brown traded)'])
ax1.set_ylabel('Slot WR EPA/Target'); ax1.set_title('Alpha WR1 Effect'); ax1.axhline(y=0, color='gray', linewidth=0.5)

# Personnel
for i, pers in enumerate(['11', '12', '13']):
    sub = shan_slot[shan_slot['personnel_code'] == pers]
    if len(sub) > 50:
        label = {'11': '11\n(3WR)', '12': '12\n(2TE)', '13': '13\n(3TE)'}.get(pers, pers)
        ax2.bar(i, sub['epa'].mean(), color=[SHANAHAN_COLOR, EAGLES_GREEN, GOLD][i], edgecolor='white', width=0.6)
        ax2.text(i, sub['epa'].mean() + 0.02, f"{sub['epa'].mean():+.3f}\nn={len(sub):,}",
                ha='center', fontsize=9, fontweight='bold')
ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(['11\n(3WR)', '12\n(2TE)', '13\n(3TE)'])
ax2.set_ylabel('Slot WR EPA/Target'); ax2.set_title('Personnel Effect'); ax2.axhline(y=0, color='gray', linewidth=0.5)

# Location
for i, loc in enumerate(['left', 'middle', 'right']):
    sub = shan_slot[shan_slot['pass_location'] == loc]
    if len(sub) > 50:
        color = EAGLES_GREEN if loc == 'middle' else OTHER_COLOR
        ax3.bar(i, sub['epa'].mean(), color=color, edgecolor='white', width=0.6)
        ax3.text(i, sub['epa'].mean() + 0.02, f"{sub['epa'].mean():+.3f}\nn={len(sub):,}",
                ha='center', fontsize=9, fontweight='bold')
ax3.set_xticks([0, 1, 2]); ax3.set_xticklabels(['Left', 'Middle', 'Right'])
ax3.set_ylabel('Slot WR EPA/Target'); ax3.set_title('Pass Location'); ax3.axhline(y=0, color='gray', linewidth=0.5)

plt.suptitle('Slot WR Efficiency Drivers (Shanahan-Tree Offenses)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '05_efficiency_drivers.png'); plt.close()
print("  Saved: 05_efficiency_drivers.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║  PART 5: MONTE CARLO PROJECTION                           ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("PART 5: MONTE CARLO PROJECTION — Lemon Season Simulations")
print("=" * 60)

# Reference play menu
shan_slot_plays = df[(df['is_shanahan'] == 1) & (df['air_yards_clean'] < 10) & (df['season'] >= 2020)][FEATURE_COLS].copy()

# Base rates
slot_catch = df[(df['is_shanahan'] == 1) & (df['air_yards_clean'] < 10) & (df['season'] >= 2020)]['complete_pass'].mean()
slot_ypc = df[(df['is_shanahan'] == 1) & (df['air_yards_clean'] < 10) & (df['complete_pass'] == 1) & (df['season'] >= 2020)]['yards_gained'].mean()
slot_yac = df[(df['is_shanahan'] == 1) & (df['air_yards_clean'] < 10) & (df['complete_pass'] == 1) & (df['season'] >= 2020)]['yards_after_catch'].mean()
slot_td = df[(df['is_shanahan'] == 1) & (df['air_yards_clean'] < 10) & (df['season'] >= 2020)]['pass_touchdown'].mean()

# Lemon-adjusted rates
adj_ypc = slot_ypc * lemon_mult
adj_catch = np.clip(slot_catch * (lemon_mult ** 0.3), 0.58, 0.78)
adj_yac = slot_yac * (lemon_mult ** 0.5)

print(f"  Quality multiplier: {lemon_mult:.2f}x")
print(f"  Adjusted rates: catch={adj_catch:.3f}, ypc={adj_ypc:.1f}, yac={adj_yac:.1f}")

def simulate_season(scenario):
    n_targets = np.random.randint(scenario['targets'][0], scenario['targets'][1] + 1)
    indices = np.random.choice(len(shan_slot_plays), size=n_targets, replace=True)
    plays = shan_slot_plays.iloc[indices].values.copy().astype(np.float32)
    for col_name, fn in scenario.get('adjustments', {}).items():
        plays[:, FEATURE_COLS.index(col_name)] = fn(plays[:, FEATURE_COLS.index(col_name)], n_targets)
    plays_s = scaler_epa.transform(plays)
    with torch.no_grad():
        mu, lv = model(torch.FloatTensor(plays_s).to(DEVICE))
        std = torch.exp(0.5 * lv)
        epa = (mu + torch.randn_like(std) * std).cpu().numpy().flatten()
    cr = np.clip(adj_catch + np.random.normal(0, 0.025), 0.55, 0.80)
    ypc = np.clip(adj_ypc + np.random.normal(0, 1.2), 7, 16)
    rec = int(n_targets * cr)
    return {'targets': n_targets, 'receptions': rec, 'yards': int(rec * ypc),
            'tds': int(n_targets * np.clip(slot_td + np.random.normal(0, 0.01), 0.01, 0.08)),
            'total_epa': float(epa.sum()), 'epa_per_target': float(epa.mean())}

SCENARIOS = [
    {'name': 'With A.J. Brown', 'targets': (80, 95), 'color': EAGLES_GREEN,
     'adjustments': {'is_screen': lambda x, n: np.random.binomial(1, 0.12, n).astype(float),
                     'air_yards_clean': lambda x, n: np.clip(x * 0.9, -5, 30),
                     'has_motion': lambda x, n: np.random.binomial(1, 0.58, n).astype(float),
                     'is_shanahan': lambda x, n: np.ones(n)}},
    {'name': 'Without A.J. Brown', 'targets': (100, 120), 'color': HIGHLIGHT,
     'adjustments': {'is_screen': lambda x, n: np.random.binomial(1, 0.08, n).astype(float),
                     'air_yards_clean': lambda x, n: np.clip(x * 1.05, -5, 35),
                     'has_motion': lambda x, n: np.random.binomial(1, 0.55, n).astype(float),
                     'is_shanahan': lambda x, n: np.ones(n)}},
]

print(f"\n  Running {N_SIMULATIONS} simulations per scenario...")
results = {}
for sc in SCENARIOS:
    sims = [simulate_season(sc) for _ in range(N_SIMULATIONS)]
    results[sc['name']] = pd.DataFrame(sims)
    s = results[sc['name']]
    print(f"\n  {sc['name']}:")
    print(f"    Yards: {s['yards'].median():.0f} (10th: {s['yards'].quantile(0.1):.0f}, 90th: {s['yards'].quantile(0.9):.0f})")
    print(f"    Rec:   {s['receptions'].median():.0f} | TDs: {s['tds'].median():.0f} | EPA/tgt: {s['epa_per_target'].median():.3f}")

# Projection figures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for metric, label, ax in [('yards', 'Receiving Yards', axes[0, 0]), ('receptions', 'Receptions', axes[0, 1]),
                            ('tds', 'Touchdowns', axes[1, 0]), ('epa_per_target', 'EPA/Target', axes[1, 1])]:
    for sc in SCENARIOS:
        data = results[sc['name']][metric]
        ax.hist(data, bins=30, alpha=0.5, color=sc['color'], label=sc['name'], density=True, edgecolor='white')
        med = data.median()
        ax.axvline(x=med, color=sc['color'], linestyle='--', linewidth=2, alpha=0.8)
        fmt = f'{med:.0f}' if metric != 'epa_per_target' else f'{med:.3f}'
        ax.annotate(fmt, xy=(med, ax.get_ylim()[1] * 0.85), fontsize=9, fontweight='bold', color=sc['color'], ha='center')
    ax.set_xlabel(label); ax.set_ylabel('Density'); ax.set_title(f'Projected {label}'); ax.legend(fontsize=9)
plt.suptitle(f"Makai Lemon: Rookie Projections ({N_SIMULATIONS:,} simulations)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '06_projections.png'); plt.close()
print("  Saved: 06_projections.png")

# Probability thresholds
fig, ax = plt.subplots(figsize=(12, 6))
thresholds = [400, 500, 600, 700, 800, 900, 1000, 1100]
for sc in SCENARIOS:
    probs = [(results[sc['name']]['yards'] >= t).mean() for t in thresholds]
    ax.plot(thresholds, probs, marker='o', linewidth=2.5, color=sc['color'], label=sc['name'], markersize=8)
    for t, p in zip(thresholds, probs):
        if t in [600, 800, 1000]:
            ax.annotate(f'{p:.0%}', xy=(t, p), textcoords='offset points', xytext=(0, 12),
                        fontsize=9, ha='center', fontweight='bold', color=sc['color'])
ax.set_xlabel('Receiving Yards'); ax.set_ylabel('Probability'); ax.set_title('Probability of Reaching Yardage Milestones')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / '07_probability_thresholds.png'); plt.close()
print("  Saved: 07_probability_thresholds.png")


# ═══════════════════════════════════════════════════════════
# EXPORT JSON FOR FRONTEND
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPORT: Frontend JSON")
print("=" * 60)

frontend = {}
for sc in SCENARIOS:
    name = sc['name']
    s = results[name]
    frontend[name] = {
        'summary': {stat: {'median': float(s[stat].median()), 'p10': float(s[stat].quantile(0.1)),
                           'p90': float(s[stat].quantile(0.9)), 'mean': float(s[stat].mean())}
                    for stat in ['targets', 'receptions', 'yards', 'tds', 'epa_per_target']},
        'distributions': {stat: s[stat].tolist() for stat in ['yards', 'receptions', 'tds']},
        'yard_thresholds': {str(t): float((s['yards'] >= t).mean()) for t in thresholds},
    }

frontend['lemon_profile'] = {
    'name': 'Makai Lemon', 'draft': {'round': 1, 'pick': 20, 'team': 'PHI'},
    'measurables': {k: v for k, v in LEMON.items() if k in ['height_inches', 'weight', 'forty', 'hand_size', 'arm_length']},
    'quality_multiplier': round(float(lemon_mult), 3),
    'comps': [{'name': r['name'], 'yr1_yards': float(r['yr1_yards']), 'career_yards': float(r['total_yards'])}
              for _, r in comps.iterrows()],
}

frontend['findings'] = {
    'motion_lift_slot_shanahan': round(float(lifts[0]), 3) if lifts else None,
    'shanahan_motion_rate': round(float(shan_motion), 3),
    'middle_of_field_epa': round(float(shan_slot[shan_slot['pass_location'] == 'middle']['epa'].mean()), 3),
    'twelve_personnel_epa': round(float(shan_slot[shan_slot['personnel_code'] == '12']['epa'].mean()), 3),
}

frontend['metadata'] = {
    'model': 'PyTorch EPA (Gaussian NLL) + Ridge Quality Adjustment',
    'epa_model_r2': round(float(final_r2), 4),
    'quality_model_alpha': round(float(ridge_cv.alpha_), 1),
    'n_simulations': N_SIMULATIONS,
    'data': 'nflverse 2016-2025 + MockDraftable measurables',
}

with open('outputs/lemon_projections.json', 'w') as f:
    json.dump(frontend, f, indent=2)
print("  Saved: outputs/lemon_projections.json")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFigures: {OUTPUT_DIR}/")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  {f.name}")
print(f"\nFrontend data: outputs/lemon_projections.json")
print(f"Model weights: outputs/best_epa_model.pt")
print("=" * 60)