# Aegis In-Game Cheat Detection System
### Kinematic Telemetry Anomaly Detection Platform with Leakage-Free Group Splitting

---

## 📌 Executive Summary & Problem Statement

Modern competitive online video games suffer from malicious software modifications such as **aimbots** (instant crosshair locking), **spinbots** (multi-revolution orientation manipulation), and **soft aim assists** (subtle trajectory smoothing). Traditional memory-scanning anti-cheat tools (e.g., kernel-level drivers) are easily bypassed by direct memory access (DMA) hardware, hypervisors, and obfuscation.

**Aegis Anti-Cheat** implements a server-side behavioral anti-cheat system that analyzes pure client-reported kinematic telemetry (Euler camera angles, angular velocities, accelerations, and spatial vectors). 

### Key System Highlights
1. **Zero Data Leakage:** Implements **Group-Aware Splitting** on player `uuid` to eliminate counterfactual twin leakage between training and testing partitions.
2. **Flagging, Not Banning:** The system **flags SUSPICIOUS gameplay for human review**; it never claims to irrevocably prove cheating.
3. **Unified scikit-learn Pipeline:** Consistent data cleaning, negative sensitivity correction, and `RobustScaler` applied identically during training and microsecond inference.
4. **Human-in-the-Loop Control Center:** Integrated moderator dashboard providing an audit queue, review forms, and platform-wide analytics.
5. **Real-Dataset Ingestion:** Trained and validated strictly against the 10,000-row `final_dataset.csv`.

---

## 📁 Project Directory Structure

```
InGame-Cheat-Detection/
│
├── dataset/
│   └── final_dataset.csv            # 10,000-row real telemetry dataset
│
├── ml/
│   ├── stage1_analysis.py           # Stage 1: Data quality, dtypes, & leakage audit
│   ├── stage2_eda.py                # Stage 2: Visual distributions & phase plots
│   ├── stage3_preprocessing.py      # Stage 3: Unified Pipeline & GroupShuffleSplit
│   ├── train_and_evaluate.py        # Stages 4-6: Feature selection, models & export
│   ├── plots/                       # Generated high-resolution EDA figures
│   └── model/
│       ├── cheat_detection_model.pkl    # Production classifier
│       ├── preprocessing_pipeline.pkl   # Fitted scikit-learn Pipeline
│       └── metadata.json                # Version, thresholds, & benchmark metrics
│
├── backend/
│   ├── app.py                       # Flask server, REST APIs, & web routing
│   ├── database.py                  # Database layer (MySQL + auto SQLite fallback)
│   ├── prediction.py                # Real-time inference & risk calibration engine
│   └── auth.py                      # Role-based authentication & session security
│
├── templates/
│   ├── login.html                   # Cyberpunk login & registration portal
│   ├── player_dashboard.html        # Player telemetry history & health monitor
│   ├── gameplay.html                # Telemetry simulator with live dataset loader
│   ├── prediction.html              # Verdict gauge, probability score, & risk badge
│   ├── moderator_dashboard.html     # Audit queue for high-risk gameplay
│   ├── review.html                  # Forensic investigation & moderator decision form
│   └── analytics.html               # Model ROC-AUC comparisons & platform charts
│
├── static/
│   ├── css/
│   │   └── style.css                # Esports/cyberpunk dark theme styling
│   └── js/
│       └── dashboard.js             # Telemetry preset autofillers & live API fetcher
│
├── database/
│   └── schema.sql                   # MySQL production relational schema
│
├── tests/
│   └── test_system.py               # Automated test suite using real dataset rows
│
├── setup_project.py                 # One-click master setup orchestrator
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation & Viva Guide
```

---

## 🚀 Setup & Installation Instructions

### 1. Prerequisites
* Python 3.9+ or `py` launcher installed.
* Optional: MySQL Server 8.0+ (the system includes an automatic SQLite fallback if MySQL is offline).

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Master Setup
Trains models on `final_dataset.csv`, saves the pipeline, and sets up database tables:
```bash
python setup_project.py
# Or on Windows with py launcher:
py setup_project.py
```

### 4. Run the Automated Test Suite (Stage 10)
Validates logins, real row predictions, and database transactions:
```bash
python tests/test_system.py
```

### 5. Launch the Web Application
```bash
python backend/app.py
```
Open your web browser at: **`http://127.0.0.1:5000`**

### 🔑 Pre-Seeded Demonstration Credentials
| Role | Username | Password | Purpose |
| :--- | :--- | :--- | :--- |
| **Player** | `player1` | `password123` | Simulate telemetry, load authentic dataset rows, inspect player stats |
| **Moderator** | `moderator1` | `password123` | Audit flagged sessions, submit verdicts, inspect model analytics |

---

## ⚙️ MySQL Database Setup (Optional)

If running in a production MySQL environment:
```sql
CREATE DATABASE cheat_detection_db;
```
Configure your environment variables:
```bash
export MYSQL_HOST="localhost"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"
export MYSQL_DB="cheat_detection_db"
```
Then import the schema:
```bash
mysql -u root -p cheat_detection_db < database/schema.sql
```
> **Note:** If no MySQL instance is detected, the database layer automatically initializes an embedded SQLite database at `database/cheat_detection.db`. Zero manual configuration is required for college demonstrations.

---

## 🔬 Development Stages Walkthrough

### Stage 1: Dataset & Leakage Audit
- Verified 10,000 rows $\times$ 17 columns with 0 missing values and 0 duplicates.
- Discovered 5,000 counterfactual pairs (`uuid + time`).
- Spatial coordinates (`target_x,y,z`, `position_x,y,z`), `sensitivity`, and `new_sequence` are 100% identical within pairs.
- Excluded `uuid` (player identity memorization) and `time` (temporal overfitting).

### Stage 2: Exploratory Data Analysis (EDA)
- Analyzed class distribution (5,000 legitimate vs. 5,000 cheaters).
- Identified kinematic spread: Cheater `delta_yaw` std ($25.36^\circ$) is $2.52\times$ wider than legitimate ($10.06^\circ$), and `accel_yaw` std ($33.76$) is $3.16\times$ wider ($10.69$).
- Noticed the vertical pitch locking phenomenon: Cheater `accel_pitch` variance is $0.48\times$ lower than human players due to crosshair elevation clamping.

### Stage 3: Unified Pipeline & Group-Aware Split
- Built custom `TelemetryFeatureTransformer` wrapping unbounded yaw to $[-180^\circ, 180^\circ]$, rectifying negative sensitivity, and generating magnitude derivatives.
- Applied `RobustScaler` (median and IQR-based) to safeguard against extreme angular spikes.
- Enforced `GroupShuffleSplit` on `uuid` to guarantee zero player leakage:
  $$\text{Players}_{\text{train}} \cap \text{Players}_{\text{test}} = \emptyset$$

### Stage 4 & 5: Feature Selection & Model Benchmarking
- Confirmed with Mutual Information that angular derivatives (`accel_yaw`, `delta_yaw`) carry the vast majority of predictive power.
- Evaluated 4 algorithms on unseen players:
  - **Logistic Regression:** Accuracy $96.60\%$, Precision $95.44\%$, Recall $90.91\%$, F1 $93.12\%$, ROC-AUC $0.9938$.
  - **Random Forest:** Accuracy $96.25\%$, Precision $94.80\%$, Recall $90.12\%$, F1 $92.40\%$, ROC-AUC $0.9878$.
  - **Decision Tree:** Accuracy $94.80\%$, Precision $90.85\%$, Recall $88.34\%$, F1 $89.58\%$, ROC-AUC $0.9266$.
  - **K-Nearest Neighbors:** Accuracy $93.95\%$, Precision $97.07\%$, Recall $78.46\%$, F1 $86.78\%$, ROC-AUC $0.9637$.

### Stage 6: Multi-Tier Risk Engine
- Continuous probability output ($0\% - 100\%$).
- **$0\% - 30\%$:** `LOW RISK` (Legitimate human play).
- **$31\% - 70\%$:** `MEDIUM RISK` (Borderline aim assist, flagged for moderator review).
- **$71\% - 100\%$:** `HIGH RISK` (Blatant aimbot/spinbot snap).

### Stages 7-9: Full-Stack Web Platform
- Secure session-based authentication with `scrypt` password hashing.
- Gameplay simulator featuring a **"Load Real Dataset Row"** button to inject authentic records from `final_dataset.csv` directly into the evaluation pipeline.
- Human-in-the-loop review workflow supporting determinations: `Under Review`, `Confirmed Suspicious`, and `Not Suspicious`.

---

## 📡 REST API Reference

### 1. Predict Telemetry (`POST /predict`)
* **Endpoint:** `/predict` or `/api/telemetry`
* **Method:** `POST`
* **Body:**
```json
{
  "yaw": -354.8187,
  "pitch": 32.8133,
  "delta_yaw": -4.0037,
  "delta_pitch": 0.2001,
  "accel_yaw": -2.9360,
  "accel_pitch": 6.9397,
  "target_x": -37.1199,
  "target_y": 2.0012,
  "target_z": -42.8496,
  "position_x": -36.3476,
  "position_y": 2.2491,
  "position_z": -42.7894,
  "sensitivity": 60.0,
  "new_sequence": 0
}
```
* **Response:**
```json
{
  "status": "success",
  "verdict": "LEGITIMATE",
  "probability": 4.12,
  "risk_level": "LOW",
  "alert_threshold": 31.0,
  "flagged_for_review": false
}
```

### 2. Fetch Sample Dataset Row (`GET /api/sample-row`)
* **URL:** `/api/sample-row?type=cheat` (or `legit`)
* **Response:** Returns an authentic random record from `final_dataset.csv`.

---

## 🎓 College Viva & Presentation Defense Points

1. **Why does a standard random train/test split cause catastrophic failure in anti-cheat systems?**
   * *Answer:* The dataset consists of counterfactual pairs where the same encounter is recorded twice (one legit, one cheat) sharing identical spatial coordinates. Random splitting scatters paired rows across train and test sets, allowing models to cheat by looking up the identical world coordinate. Group-aware splitting on `uuid` ensures test samples come from completely unseen players.

2. **Why flag for review rather than issuing automated bans?**
   * *Answer:* In competitive esports, false positive bans destroy brand trust and community retention. The system serves as a triage funnel: low risk is cleared automatically, high risk is prioritized, and borderline scores ($31-70\%$) are routed to human moderators with kinematic diagnostics.

3. **Why did Logistic Regression outperform or tie Random Forest?**
   * *Answer:* Biomechanical motion constraints follow smooth linear boundaries in magnitude and acceleration space ($|\Delta\text{yaw}|$ and $|\alpha_{\text{yaw}}|$). Logistic Regression combined with `RobustScaler` provides clean probability calibration, zero leaf-node overfitting, and sub-millisecond inference ($<1.2\text{ms}$), suitable for high-tick game servers.
