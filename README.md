# EVC & Ghost Traffic Attribution Analyzer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Benfinkels/Cross-Channel-Attribution-Analyzer-EVC-Impact-Model/blob/main/EVC_Session_Attribution.ipynb)

A statistical modeling suite that quantifies the incremental cross-channel impact of YouTube **Engaged-View Conversions (EVCs)** on high-intent channels (Direct, Organic Search, Paid Search) using **Time-Lagged Non-Negative Least Squares (NNLS)** and **Ridge Regression**.

---

## 1. The Attribution Problem

Video advertising (YouTube) creates top-of-funnel demand that rarely manifests as an immediate click. Users watch a video ad, consider the brand, and later visit the site via high-intent channels such as:
- **Direct navigation**
- **Organic Search (Brand)**
- **Paid Search (CPC)**

Standard analytics platforms (like Google Analytics 4) credit the *last non-direct click*, leaving Video campaigns appearing inefficient while "free" or bottom-funnel channels appear artificially inflated.

**The Solution:** This tool mathematically "unmixes" site traffic across channels. It scans time lags between 0 and 14 days and detects which channels statistically rise and fall in correlation with video view spikes, effectively tracing "Ghost Traffic" back to its true acquisition origin.

---

## 2. Methodology: Dual-Model Approach

| Model | Nickname | Statistical Method | Best For | Interpretation |
|---|---|---|---|---|
| **Model A** | *The Judge* | **Non-Negative Least Squares (NNLS)** | Identifying the **Primary Driver** | Conservative & strict. Forces channels to compete for credit ($w_i \ge 0$). |
| **Model B** | *The Diplomat* | **Ridge Regression (L2 penalty)** | Measuring the **Halo Effect** | Captures multi-channel lift and collinearity across channels. |

### Auto-Lag Detection
Rather than assuming an arbitrary lookback window, the tool automatically tests lags from $t=0$ to $t=14$ days and locks onto the delay that optimizes predictive fit ($R^2$).

---

## 3. Quickstart

### Option A: Run in Google Colab (One-Click)
Click the badge above or open [`EVC_Session_Attribution.ipynb`](EVC_Session_Attribution.ipynb) directly in Google Colab.

### Option B: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Benfinkels/Cross-Channel-Attribution-Analyzer-EVC-Impact-Model.git
   cd Cross-Channel-Attribution-Analyzer-EVC-Impact-Model
   ```

2. **Set up a Python environment (Python 3.10+):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Launch Jupyter Lab or Notebook:**
   ```bash
   jupyter lab
   ```
   Open `EVC_Session_Attribution.ipynb` and run the cells. Ready-to-use sample datasets (`sample_ga4_sessions.csv` and `sample_google_ads_evc.csv`) are bundled in the repository so the entire analysis runs immediately out of the box.

---

## 4. Using Your Own Data

The notebook includes a smart auto-parser that accepts GA4 and Google Ads exports in **Long**, **Wide**, or **Pivot** formats:

1. **Session Data (Effect):** Export from GA4 with:
   - `Date` (e.g. `2024-01-15`)
   - `Session source / medium` or `Channel`
   - `Sessions`
2. **Ad Data (Cause):** Export from Google Ads with:
   - `Date`
   - `Engaged-view conversions` or `Conversions`

Upload your CSVs using the interactive loader or drop them into the repository folder.

---

## 5. Outputs & Deliverables

- **`EVC_Competitive_Model.csv`**: Coefficients and attributed conversion counts per channel from NNLS.
- **`EVC_Competitive_Model_Ridge.csv`**: Coefficients and attributed conversions from Ridge Regression.
- **Interactive Dual Stack Plots**: Stacked time series showing predicted channel volume aligned against actual reported EVCs.
- **Ghost ROI Analysis**: Translates attributed conversions into monetary value using Average Order Value (AOV) and Target CPA.
- **`Attribution_Master_Report.html`**: A standalone, self-contained executive dashboard report ready to share with stakeholders.

---

## License

MIT License.
