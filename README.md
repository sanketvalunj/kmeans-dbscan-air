# Air Quality Zone Clustering and Pollution Anomaly Detection Using K-Means and DBSCAN

DWM (Data Warehousing and Mining) mini-project. An **unsupervised, exploratory** analysis of Indian air-quality data:

1. **City-level profiling** - group cities with similar pollutant profiles using K-Means (K guided by the Elbow Method).
2. **Observation-level analysis** - cluster daily city observations with K-Means, then use DBSCAN to find density-based groups and *potential* anomalies (noise points).

> Cluster IDs are **not** official air-quality categories, and DBSCAN noise points are **not** confirmed pollution emergencies. They are starting points for investigation.

**Execution status of this package:** the code was written and checked for errors using a *synthetic stand-in CSV* only. It has **not** been run on the real Kaggle dataset (which was not available when the project was generated). No result images, CSVs, scores or cluster counts are included - all are produced when *you* run the project. Run it on the real data before writing any conclusions.

---

## Project structure

```
Air_Quality_Zone_Clustering_Project/
├── Air_Quality_Zone_Clustering.ipynb   # main demonstration notebook (no stored outputs)
├── main.py                             # standalone script, same workflow
├── requirements.txt
├── README.md
├── PROJECT_FILE_MANIFEST.md
├── DWM_Mini_Project_Report.docx        # academic report (results are placeholders)
├── data/raw/                           # put city_day.csv here
└── outputs/                            # created/filled when the code runs
```

## 1. Get the dataset (manual)

1. Open <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india> (free Kaggle login needed).
2. Download the archive and extract **`city_day.csv`**.
3. Put it at exactly: **`data/raw/city_day.csv`**

(Optional, if you have the Kaggle CLI configured: `kaggle datasets download -d rohanrao/air-quality-data-in-india -p data/raw --unzip`)

## 2. Set up (macOS, Apple Silicon, VS Code)

Open the project folder in VS Code, open the built-in terminal, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` already includes JupyterLab and ipykernel. If you removed them: `pip install jupyterlab ipykernel`.

**Select the interpreter in VS Code:** press `Cmd+Shift+P` -> *Python: Select Interpreter* -> choose the one inside `.venv` (`./.venv/bin/python`). For the notebook, click **Select Kernel** (top right) -> *Python Environments* -> `.venv`.

## 3. Run

**Notebook (recommended for demonstration):**
```bash
jupyter lab        # or: jupyter notebook
```
Open `Air_Quality_Zone_Clustering.ipynb` and choose *Run All*. (Start Jupyter from the project folder so the relative data path works. In VS Code you can also just open the `.ipynb` and click *Run All*.)

**Script:**
```bash
python main.py
```
`main.py` saves figures to files without opening windows.

## 4. Expected outputs (in `outputs/`)

| File | Contents |
|---|---|
| `city_cluster_results.csv` | each city, its median pollutants (original units), days of data, median AQI, `city_cluster` |
| `city_cluster_profiles.csv` | per-cluster median pollutants, size, member cities |
| `daily_kmeans_results.csv` | daily rows with metadata, original-unit pollutants and `kmeans_cluster` |
| `daily_cluster_profiles.csv` | per-cluster median pollutants, size, AQI info |
| `dbscan_results_sample.csv` | sampled rows with DBSCAN label, noise flag, k-distance |
| `potential_pollution_anomalies.csv` | only the DBSCAN noise rows (potential anomalies, sorted by isolation) |
| `run_summary.json` | all key numbers (shape, K values, silhouette, PCA variance, DBSCAN counts) for your report |
| `missing_values.png`, `city_elbow.png`, `daily_elbow.png`, `city_kmeans_pca.png`, `daily_kmeans_pca.png`, `dbscan_k_distance.png`, `dbscan_pca.png` | figures (200 dpi) |

## 5. Settings you can change

All settings are in one block (`CONFIGURATION` in `main.py`, the "Imports and Configuration" section in the notebook):

* `CITY_K`, `DAILY_K` - `None` uses the elbow suggestion; set an integer to override.
* `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES`, `DBSCAN_MAX_SAMPLES` - DBSCAN radius, density threshold and sample cap (default 6,000 rows).
* `MIN_FEATURE_COVERAGE`, `MIN_FEATURES_PER_ROW` - how sparse a pollutant column or a row may be before it is dropped.
* `RANDOM_STATE` - seed (42).

## 6. Method summary

Load -> validate columns -> convert to numeric -> negatives to NaN -> drop sparse rows/columns -> median imputation -> `log1p` -> `StandardScaler` -> Elbow + K-Means (city level, daily level) -> silhouette (sampled if large) -> k-distance plot -> DBSCAN on a reproducible sample -> PCA plots -> CSV/PNG export.
Metadata (`City`, `Date`, `AQI`, `AQI_Bucket`) is **never** a clustering feature; AQI is derived from pollutants, so it is kept only for interpretation.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Put `city_day.csv` at `data/raw/city_day.csv`; run from the project folder. |
| `ModuleNotFoundError` | Activate `.venv` (`source .venv/bin/activate`) then `pip install -r requirements.txt`. In VS Code make sure the `.venv` interpreter/kernel is selected. |
| `Column 'City' not found` / no pollutant columns | You used a different file (e.g. `station_day.csv`). Use `city_day.csv`, or edit the column names in the configuration block. |
| Some pollutants missing from results | They had less than 30% usable data and were dropped on purpose (lower `MIN_FEATURE_COVERAGE` to keep them). |
| DBSCAN: every point is noise | `eps` too small or `min_samples` too large - raise `DBSCAN_EPS` or lower `DBSCAN_MIN_SAMPLES`. |
| DBSCAN: one big cluster, almost no noise | `eps` too large - lower `DBSCAN_EPS`. |
| `eps must be > 0` | Set `DBSCAN_EPS` to a positive number. |
| Plots do not appear when running `main.py` | Intended - they are saved in `outputs/`. |
| Jupyter cannot find the file | Start `jupyter lab` from the project root, not from a sub-folder. |

## 8. References

* Dataset: Rohan Rao, *Air Quality Data in India (2015 - 2020)*, Kaggle. <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>
* scikit-learn `KMeans`: <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
* scikit-learn `DBSCAN`: <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>
* scikit-learn `StandardScaler`: <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>
* scikit-learn `PCA`: <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>
* scikit-learn `silhouette_score`: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html>
* scikit-learn clustering user guide: <https://scikit-learn.org/stable/modules/clustering.html>
# kmeans-dbscan-air
