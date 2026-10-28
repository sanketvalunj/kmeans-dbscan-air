"""
Air Quality Zone Clustering and Pollution Anomaly Detection
Using K-Means (Elbow Method) and DBSCAN
-----------------------------------------------------------
DWM Mini-Project | Dataset: "Air Quality Data in India" (Rohan Rao, Kaggle)
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

How to run (from the project root, with the virtual environment active):
    python main.py

Expected input : data/raw/city_day.csv   (download it manually from Kaggle)
Outputs        : outputs/*.csv, outputs/*.png, outputs/run_summary.json

IMPORTANT: This is an UNSUPERVISED, EXPLORATORY analysis. Cluster IDs are not
official air-quality categories and DBSCAN "noise" points are only *potential*
unusual observations that deserve a closer look - not confirmed emergencies.
"""

# ---- SECTION: imports ----
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # [MAIN_ONLY] save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
PROJECT_ROOT = Path(__file__).resolve().parent  # [MAIN_ONLY]
SHOW_PLOTS = False  # [MAIN_ONLY]

# ---- SECTION: config ----
# =============================================================================
# CONFIGURATION - every tunable setting lives here. Change values and re-run.
# =============================================================================
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "city_day.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATASET_URL = "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"

# Column names (the code checks which of these really exist in the CSV)
CITY_COL, DATE_COL, AQI_COL, BUCKET_COL = "City", "Date", "AQI", "AQI_Bucket"
POLLUTANT_CANDIDATES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

RANDOM_STATE = 42            # fixed seed -> reproducible results
MIN_FEATURE_COVERAGE = 0.30  # keep a pollutant only if >= 30% of its values are usable
MIN_FEATURES_PER_ROW = 3     # keep a daily row only if it has >= 3 measured pollutants

# --- K-Means / Elbow ---------------------------------------------------------
KMEANS_N_INIT = 10           # number of random restarts for each K-Means fit
CITY_K = None                # None = use the elbow suggestion; or set an int (e.g. 3) to override
CITY_K_MAX = 8               # elbow curve tests K = 2 ... min(8, n_cities - 1)
DAILY_K = None               # None = use the elbow suggestion; or set an int (e.g. 4) to override
DAILY_K_MAX = 10             # elbow curve tests K = 2 ... DAILY_K_MAX

# --- Silhouette / plotting ---------------------------------------------------
SILHOUETTE_SAMPLE_SIZE = 10000   # silhouette on large data is scored on a reproducible sample
PLOT_SAMPLE_SIZE = 15000         # max points drawn in the daily PCA scatter (speed/readability)

# --- DBSCAN ------------------------------------------------------------------
DBSCAN_MAX_SAMPLES = 6000    # DBSCAN is run on at most this many daily observations
DBSCAN_MIN_SAMPLES = 10      # min_samples (the point itself counts toward this number)
DBSCAN_EPS = None            # None = starting value from the k-distance plot; or set a float (e.g. 1.2)

# ---- SECTION: helpers_general ----
# =============================================================================
# GENERAL HELPERS
# =============================================================================
def save_figure(fig, filename):
    """Save a figure as a high-resolution PNG in outputs/, show it if allowed, then close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  saved figure -> outputs/{filename}")


def save_csv(df, filename, index=False):
    """Save a DataFrame as CSV in outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=index)
    print(f"  saved table  -> outputs/{filename}  ({len(df)} rows)")


def to_builtin(obj):
    """Convert numpy types so the summary can be written as JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj

# ---- SECTION: loading ----
# =============================================================================
# 1. DATA LOADING AND EXPLORATION
# =============================================================================
def load_dataset(csv_path):
    """Read the CSV. Stops with a clear message if the file is missing."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            "\nDataset not found.\n"
            f"  Expected location : {csv_path}\n"
            f"  Download from     : {DATASET_URL}\n"
            "  Use the file 'city_day.csv' and place it at data/raw/city_day.csv\n"
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()          # remove accidental spaces in headers
    if CITY_COL not in df.columns:
        raise ValueError(f"Column '{CITY_COL}' not found. Available columns: {list(df.columns)}")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def inspect_dataset(df):
    """Print basic facts about the loaded data (nothing is hard-coded)."""
    print("Shape (rows, columns):", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nNumber of distinct cities:", df[CITY_COL].nunique())
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        print("Date range:", df[DATE_COL].min().date(), "to", df[DATE_COL].max().date())
        print("Rows with unparseable dates:", int(df[DATE_COL].isna().sum()))
    print("\nMissing values per column:\n", df.isna().sum())

# ---- SECTION: missing ----
# =============================================================================
# 2. MISSING-VALUE ANALYSIS
# =============================================================================
def missing_percent_table(df, columns):
    """Percentage of missing values for the given columns (after numeric conversion)."""
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return (numeric.isna().mean() * 100).round(2)


def plot_missing_values(raw_pct, clean_pct, filename="missing_values.png"):
    """Bar chart comparing missing % before and after cleaning."""
    table = pd.DataFrame({
        "Before cleaning": raw_pct,
        "After cleaning (negatives -> NaN, sparse rows dropped)": clean_pct,
    })
    table = table.rename_axis("Pollutant").reset_index()
    long = table.melt(id_vars="Pollutant", var_name="Stage", value_name="Missing (%)")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="Pollutant", y="Missing (%)", hue="Stage", ax=ax)
    ax.set_title("Missing values per pollutant")
    ax.set_ylabel("Missing values (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Stage")
    save_figure(fig, filename)

# ---- SECTION: preprocessing ----
# =============================================================================
# 3. VALIDATION AND PREPROCESSING
# =============================================================================
def select_pollutant_columns(df):
    """Pick pollutant columns that exist AND contain enough usable numeric data."""
    present = [c for c in POLLUTANT_CANDIDATES if c in df.columns]
    if not present:
        raise ValueError("None of the expected pollutant columns were found. "
                         f"Expected some of {POLLUTANT_CANDIDATES}; found {list(df.columns)}")
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean()
    selected = [c for c in present if coverage[c] >= MIN_FEATURE_COVERAGE]
    dropped = [c for c in present if c not in selected]
    absent = [c for c in POLLUTANT_CANDIDATES if c not in df.columns]
    print("Pollutant columns present  :", present)
    print("Pollutant columns absent   :", absent)
    print(f"Dropped (coverage < {MIN_FEATURE_COVERAGE:.0%}):", dropped)
    print("Pollutant columns SELECTED :", selected)
    if len(selected) < 2:
        raise ValueError("Fewer than 2 usable pollutant columns - cannot cluster or draw PCA.")
    return selected, present


def clean_pollutants(df, pollutants):
    """
    Cleaning policy (documented):
      1. Convert pollutant columns to numbers (bad text -> NaN).
      2. Negative concentrations are physically invalid -> set to NaN (NOT passed to log1p).
      3. Drop rows with no City, and rows with fewer than MIN_FEATURES_PER_ROW measured pollutants.
      4. Drop exact duplicate City+Date rows (if Date exists).
    Metadata (City, Date, AQI, AQI_Bucket) is kept in the table but is never a clustering feature.
    """
    work = df.copy()
    work[pollutants] = work[pollutants].apply(pd.to_numeric, errors="coerce")
    if AQI_COL in work.columns:
        work[AQI_COL] = pd.to_numeric(work[AQI_COL], errors="coerce")

    neg_mask = work[pollutants] < 0
    n_negative = int(neg_mask.sum().sum())
    work[pollutants] = work[pollutants].mask(neg_mask)

    n_start = len(work)
    work = work[work[CITY_COL].notna()]
    n_enough = work[pollutants].notna().sum(axis=1) >= min(MIN_FEATURES_PER_ROW, len(pollutants))
    work = work[n_enough]
    n_duplicates = 0
    if DATE_COL in work.columns:
        before = len(work)
        work = work.drop_duplicates(subset=[CITY_COL, DATE_COL])
        n_duplicates = before - len(work)
    work = work.reset_index(drop=True)

    stats = {"rows_before_cleaning": n_start, "rows_after_cleaning": len(work),
             "rows_removed": n_start - len(work), "negative_values_set_to_nan": n_negative,
             "duplicate_city_date_rows_removed": n_duplicates}
    print("Cleaning summary:", stats)
    return work, stats


def transform_features(df, columns):
    """
    Median imputation -> log1p -> StandardScaler.
    Returns (scaled matrix, list of columns actually used, fitted scaler).
    NOTE: this is an unsupervised analysis on the full dataset (no train/test split), so
    the imputer and scaler are fitted once on the data being clustered; the same fitted
    scaler is then reused for every later step (no refitting on subsets -> no inconsistencies).
    """
    used = [c for c in columns if df[c].notna().any()]       # drop all-empty features
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(df[used])
    if (imputed < 0).any():
        raise ValueError("Negative values reached log1p - cleaning step failed.")
    logged = np.log1p(imputed)                                # tames right-skewed concentrations
    scaler = StandardScaler()
    scaled = scaler.fit_transform(logged)                     # mean 0, std 1 per feature
    return scaled, used, scaler

# ---- SECTION: elbow ----
# =============================================================================
# 4. K-MEANS HELPERS (ELBOW, SILHOUETTE, PROFILES)
# =============================================================================
def valid_k_values(n_samples, k_max):
    """K must be >= 2 and smaller than the number of samples."""
    upper = min(k_max, n_samples - 1)
    if upper < 2:
        raise ValueError(f"Not enough samples ({n_samples}) to try K >= 2.")
    return list(range(2, upper + 1))


def compute_inertias(X, k_values):
    """Inertia = sum of squared distances of points to their nearest centroid (WCSS)."""
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        model.fit(X)
        inertias.append(float(model.inertia_))
    return inertias


def suggest_k_elbow(k_values, inertias):
    """
    Heuristic 'elbow': the K whose point is farthest from the straight line joining the first
    and last points of the (normalised) curve. It is only a SUGGESTION - always look at the plot.
    """
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)
    if len(x) < 3 or np.ptp(y) == 0:
        return int(x[0])
    xn = (x - x.min()) / np.ptp(x)
    yn = (y - y.min()) / np.ptp(y)
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    distance = np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / np.hypot(dx, dy)
    return int(x[int(np.argmax(distance))])


def plot_elbow(k_values, inertias, suggested_k, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, inertias, marker="o")
    ax.axvline(suggested_k, linestyle="--", label=f"Heuristic elbow suggestion: K = {suggested_k}")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.legend()
    save_figure(fig, filename)


def choose_k(override, k_values, inertias, name):
    """Use the user's override if given (and valid); otherwise the elbow suggestion."""
    suggested = suggest_k_elbow(k_values, inertias)
    if override is not None:
        if override < 2 or override > max(k_values):
            raise ValueError(f"{name} override ({override}) must be between 2 and {max(k_values)}.")
        print(f"{name}: elbow suggested K={suggested}, but manual override K={override} is used.")
        return int(override), suggested
    print(f"{name}: using elbow-suggested K={suggested} (set {name} in the config to override).")
    return suggested, suggested


def fit_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(X)
    return model, labels


def safe_silhouette(X, labels):
    """
    Silhouette score in [-1, 1] (higher = tighter, better-separated clusters).
    Returned only when valid: at least 2 clusters and fewer clusters than observations.
    Large data are scored on a reproducible random sample.
    """
    n, n_clusters = len(labels), len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= n:
        print("Silhouette skipped: needs >= 2 clusters and fewer clusters than observations.")
        return None
    sample = SILHOUETTE_SAMPLE_SIZE if n > SILHOUETTE_SAMPLE_SIZE else None
    try:
        return float(silhouette_score(X, labels, sample_size=sample, random_state=RANDOM_STATE))
    except ValueError as err:
        print("Silhouette could not be computed:", err)
        return None


def make_profile(df, columns, label_col, list_col=None):
    """Cluster profile in ORIGINAL units: median of each pollutant, plus size (and AQI info)."""
    grouped = df.groupby(label_col)
    profile = grouped[columns].median().round(2)
    profile.insert(0, "n_members", grouped.size())
    if AQI_COL in df.columns:
        profile["AQI_median"] = grouped[AQI_COL].median().round(1)
    if BUCKET_COL in df.columns:
        profile["most_common_AQI_Bucket"] = grouped[BUCKET_COL].agg(
            lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    if list_col is not None:
        profile["members"] = grouped[list_col].agg(lambda s: ", ".join(sorted(map(str, s))))
    return profile.reset_index()

# ---- SECTION: pca ----
# =============================================================================
# 5. PCA VISUALISATION
# =============================================================================
def plot_pca_clusters(X, labels, title, filename, annotate=None, max_points=None, noise_label=None):
    """
    Project the standardised features onto 2 principal components and colour by cluster.
    PCA is used ONLY for drawing - it does not prove that clusters are truly separable.
    Returns the explained-variance ratio of the two components.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    ratio = pca.explained_variance_ratio_

    index = np.arange(len(coords))
    if max_points is not None and len(index) > max_points:
        index = np.sort(np.random.default_rng(RANDOM_STATE).choice(index, max_points, replace=False))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for lab in np.unique(labels):
        mask = labels[index] == lab
        pts = coords[index][mask]
        if noise_label is not None and lab == noise_label:
            ax.scatter(pts[:, 0], pts[:, 1], c="black", marker="x", s=22, label=f"Noise (n={mask.sum()})")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=28 if annotate is not None else 10, alpha=0.7,
                       label=f"Cluster {lab} (n={mask.sum()})")
    if annotate is not None:                       # label each point (used for the city plot)
        for (px, py), name in zip(coords, annotate):
            ax.annotate(str(name), (px, py), fontsize=7, alpha=0.8, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%} of variance)")
    ax.set_title(f"{title}\nPCA explained variance (PC1+PC2) = {ratio.sum():.1%}")
    ax.legend(fontsize=8, loc="best")
    save_figure(fig, filename)
    return [float(r) for r in ratio]

# ---- SECTION: dbscan_helpers ----
# =============================================================================
# 6. DBSCAN HELPERS
# =============================================================================
def k_distances(X, k):
    """Distance from every point to its k-th nearest neighbour (the point itself counts as 1st)."""
    nn = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, -1]


def suggest_eps(sorted_kdist):
    """Starting value for eps: the bend ('knee') of the sorted k-distance curve."""
    y = np.asarray(sorted_kdist, dtype=float)
    if len(y) < 3 or np.ptp(y) == 0:
        return float(np.median(y))
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / np.ptp(y)
    distance = np.abs((yn[-1] - yn[0]) * (x - x[0]) - (x[-1] - x[0]) * (yn - yn[0]))
    return float(y[int(np.argmax(distance))])


def plot_k_distance(sorted_kdist, k, eps, filename="dbscan_k_distance.png"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sorted_kdist)
    ax.axhline(eps, linestyle="--", label=f"eps used = {eps:.3f}")
    ax.set_xlabel("Observations (sorted by distance)")
    ax.set_ylabel(f"Distance to the {k}-th nearest neighbour")
    ax.set_title("DBSCAN k-distance plot (look for the bend to choose eps)")
    ax.legend()
    save_figure(fig, filename)


def describe_extreme_features(X_scaled, feature_names):
    """For each row: which feature is the most extreme (largest |z|) and in which direction."""
    j = np.abs(X_scaled).argmax(axis=1)
    values = X_scaled[np.arange(len(X_scaled)), j]
    names = np.array(feature_names)[j]
    return names, np.where(values > 0, "high", "low")

# ---- SECTION: main ----
# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    summary = {}
    print("=" * 70)
    print("AIR QUALITY ZONE CLUSTERING  (K-Means + Elbow + DBSCAN)")
    print("=" * 70)

    # ---- 1. Load and explore ------------------------------------------------
    try:
        df = load_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError) as err:
        print(err)
        sys.exit(1)
    inspect_dataset(df)
    summary["raw_shape"] = list(df.shape)
    summary["n_cities_raw"] = int(df[CITY_COL].nunique())

    # ---- 2/3. Validate columns, missingness, clean ---------------------------
    print("\n--- Feature validation ---")
    pollutants, present = select_pollutant_columns(df)
    raw_missing = missing_percent_table(df, present)
    df_clean, clean_stats = clean_pollutants(df, pollutants)
    clean_missing = missing_percent_table(df_clean, pollutants).reindex(present)
    print("\nMissing % BEFORE cleaning:\n", raw_missing)
    print("\nMissing % AFTER cleaning (before imputation):\n", clean_missing)
    print("\nMissing % AFTER median imputation: 0.0 for every used feature (by construction)")
    plot_missing_values(raw_missing, clean_missing)
    summary.update({"pollutants_used": pollutants, "cleaning": clean_stats,
                    "missing_pct_before": raw_missing.to_dict(),
                    "missing_pct_after_cleaning": clean_missing.to_dict()})
    if len(df_clean) < 10:
        print("Too few valid rows after cleaning; check your dataset.")
        sys.exit(1)

    meta_cols = [c for c in [CITY_COL, DATE_COL, AQI_COL, BUCKET_COL] if c in df_clean.columns]

    # ---- 4/5. CITY-LEVEL ANALYSIS -------------------------------------------
    print("\n--- City-level K-Means ---")
    city_df = df_clean.groupby(CITY_COL)[pollutants].median()          # median: robust to spikes
    city_df = city_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    city_df["n_days"] = df_clean.groupby(CITY_COL).size().reindex(city_df.index)
    if AQI_COL in df_clean.columns:
        city_df[AQI_COL] = df_clean.groupby(CITY_COL)[AQI_COL].median().reindex(city_df.index)
    city_df = city_df.reset_index()
    city_features = [c for c in pollutants if c in city_df.columns]
    X_city, city_features, _ = transform_features(city_df, city_features)
    print("Cities clustered:", len(city_df), "| features:", city_features)

    city_ks = valid_k_values(len(city_df), CITY_K_MAX)
    city_inertias = compute_inertias(X_city, city_ks)
    city_k, city_suggested = choose_k(CITY_K, city_ks, city_inertias, "CITY_K")
    plot_elbow(city_ks, city_inertias, city_suggested, "Elbow Method - city-level profiles",
               "city_elbow.png")
    _, city_labels = fit_kmeans(X_city, city_k)
    city_df["city_cluster"] = city_labels
    city_profile = make_profile(city_df, city_features, "city_cluster", list_col=CITY_COL)
    print("\nCity cluster assignments:\n", city_df[[CITY_COL, "city_cluster"]].to_string(index=False))
    print("\nCity cluster profiles (median, original units):\n", city_profile.to_string(index=False))
    city_ratio = (plot_pca_clusters(X_city, city_labels, f"City-level K-Means (K={city_k})",
                                    "city_kmeans_pca.png", annotate=city_df[CITY_COL])
                  if X_city.shape[1] >= 2 else None)
    city_sil = safe_silhouette(X_city, city_labels)
    print("City silhouette score:", city_sil)
    save_csv(city_df.sort_values("city_cluster"), "city_cluster_results.csv")
    save_csv(city_profile, "city_cluster_profiles.csv")
    summary["city_level"] = {"n_cities": len(city_df), "k_values_tested": city_ks,
                             "inertias": city_inertias, "elbow_suggested_k": city_suggested,
                             "k_used": city_k, "silhouette": city_sil, "pca_explained_variance": city_ratio,
                             "cluster_sizes": pd.Series(city_labels).value_counts().sort_index().to_dict()}

    # ---- 6/7. DAILY OBSERVATION K-MEANS -------------------------------------
    print("\n--- Daily-observation K-Means ---")
    X_daily, daily_features, _ = transform_features(df_clean, pollutants)
    print("Daily observations:", X_daily.shape[0], "| features:", daily_features)
    daily_ks = valid_k_values(len(X_daily), DAILY_K_MAX)
    daily_inertias = compute_inertias(X_daily, daily_ks)
    daily_k, daily_suggested = choose_k(DAILY_K, daily_ks, daily_inertias, "DAILY_K")
    plot_elbow(daily_ks, daily_inertias, daily_suggested, "Elbow Method - daily observations",
               "daily_elbow.png")
    _, daily_labels = fit_kmeans(X_daily, daily_k)
    daily_out = df_clean[meta_cols + pollutants].copy()
    daily_out["kmeans_cluster"] = daily_labels
    daily_profile = make_profile(daily_out, pollutants, "kmeans_cluster")
    print("\nDaily cluster profiles (median, original units):\n", daily_profile.to_string(index=False))
    daily_ratio = plot_pca_clusters(X_daily, daily_labels, f"Daily observations K-Means (K={daily_k})",
                                    "daily_kmeans_pca.png", max_points=PLOT_SAMPLE_SIZE)
    daily_sil = safe_silhouette(X_daily, daily_labels)
    print("Daily silhouette score:", daily_sil,
          "(computed on a sample)" if len(X_daily) > SILHOUETTE_SAMPLE_SIZE else "")
    print("Note: a higher silhouette is NOT the only measure of usefulness - check the profiles too.")
    save_csv(daily_out, "daily_kmeans_results.csv")
    save_csv(daily_profile, "daily_cluster_profiles.csv")
    summary["daily_level"] = {"n_observations": int(X_daily.shape[0]), "k_values_tested": daily_ks,
                              "inertias": daily_inertias, "elbow_suggested_k": daily_suggested,
                              "k_used": daily_k, "silhouette": daily_sil,
                              "silhouette_on_sample": bool(len(X_daily) > SILHOUETTE_SAMPLE_SIZE),
                              "pca_explained_variance": daily_ratio,
                              "cluster_sizes": pd.Series(daily_labels).value_counts().sort_index().to_dict()}

    # ---- 8. DBSCAN ------------------------------------------------------------
    print("\n--- DBSCAN (sample-based) ---")
    n_all = len(X_daily)
    sample_size = min(DBSCAN_MAX_SAMPLES, n_all)
    sample_idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(n_all, sample_size, replace=False))
    X_sample = X_daily[sample_idx]
    print(f"DBSCAN is run on a reproducible random sample of {sample_size} of {n_all} observations.")
    if sample_size < 3:
        print("Sample too small for DBSCAN; skipping.")
        summary["dbscan"] = {"skipped": True}
    else:
        min_samples = min(DBSCAN_MIN_SAMPLES, sample_size)
        if min_samples < DBSCAN_MIN_SAMPLES:
            print(f"min_samples reduced to {min_samples} (sample is smaller than the configured value).")
        kdist = k_distances(X_sample, min_samples)
        sorted_kdist = np.sort(kdist)
        eps_suggested = suggest_eps(sorted_kdist)
        eps = float(DBSCAN_EPS) if DBSCAN_EPS is not None else eps_suggested
        if eps <= 0:
            raise ValueError("eps must be > 0. Set DBSCAN_EPS in the configuration section.")
        print(f"min_samples = {min_samples} | eps = {eps:.4f} "
              f"({'manual' if DBSCAN_EPS is not None else 'from k-distance bend - a starting point only'})")
        plot_k_distance(sorted_kdist, min_samples, eps)

        db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_sample)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = int((db_labels == -1).sum())
        print(f"DBSCAN clusters (excluding noise): {n_clusters}")
        print(f"Noise points (label -1): {n_noise}  ({n_noise / sample_size:.1%} of the sample)")
        if n_clusters == 0:
            print("WARNING: every point is noise -> eps is probably too small (or min_samples too large).")
        elif n_clusters == 1 and n_noise < 0.01 * sample_size:
            print("NOTE: one big cluster and almost no noise -> eps may be too large.")
        plot_pca_clusters(X_sample, db_labels, "DBSCAN on sampled daily observations",
                          "dbscan_pca.png", noise_label=-1)

        sample_out = df_clean.iloc[sample_idx][meta_cols + pollutants].copy()
        sample_out["dbscan_label"] = db_labels
        sample_out["is_noise"] = db_labels == -1
        sample_out["k_distance"] = kdist.round(4)
        extreme_name, extreme_dir = describe_extreme_features(X_sample, daily_features)
        sample_out["most_extreme_feature"] = extreme_name
        sample_out["extreme_direction"] = extreme_dir
        potential = sample_out[sample_out["is_noise"]].sort_values("k_distance", ascending=False)
        save_csv(sample_out, "dbscan_results_sample.csv")
        save_csv(potential, "potential_pollution_anomalies.csv")
        print("These are POTENTIAL unusual observations for further investigation, "
              "not confirmed pollution emergencies.")
        summary["dbscan"] = {"sample_size": sample_size, "total_observations": n_all,
                             "eps": eps, "eps_from_kdistance": eps_suggested,
                             "min_samples": min_samples, "n_clusters_excluding_noise": n_clusters,
                             "n_noise": n_noise, "noise_fraction": n_noise / sample_size}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(to_builtin(summary), f, indent=2)
    print("\nDone. All results are in the 'outputs/' folder (see run_summary.json for key numbers).")


if __name__ == "__main__":
    main()
