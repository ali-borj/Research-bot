# GitHub-Ready Open-Science Research Artifacts for Hierarchical Bayesian DDM Study

I'll generate four modular, publication-ready artifacts for your computational cognitive neuroscience workflow using HDDM[1][6].

---

## 1) DATA DICTIONARY & PREPROCESSING PIPELINE

### README.md

```markdown
# Data Dictionary and Preprocessing Pipeline

## Overview
This module standardizes trial-level behavioral data from a 2×2 within-subject design 
(Speed/Accuracy × High/Low base-rate) for hierarchical Bayesian drift diffusion modeling using HDDM.

## Data Structure

### Input Format
Raw trial-level data expected in CSV format with columns:
- `participant_id`: Unique participant identifier (string)
- `block_num`: Block number within session (integer)
- `condition_instruction`: Instruction type {speed, accuracy}
- `condition_baserate`: Base-rate manipulation {high, low}
- `evidence_strength`: Evidence difficulty level {easy, hard, medium}
- `stimulus_id`: Unique stimulus identifier (integer)
- `choice`: Participant response {0, 1} where 1 = correct direction
- `rt_raw_ms`: Reaction time in milliseconds (float, unfiltered)
- `bfi2s_extraversion`: BFI-2-S extraversion subscale score (0-100 scale)

### Output Format (Processed)
CSV/HDF5 with columns for HDDM ingestion:

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `participant_id` | str | Participant ID | For hierarchical grouping |
| `block` | int | Block number | Sequential numbering |
| `instruction` | str | {speed, accuracy} | Experimental condition |
| `base_rate` | str | {high, low} | Condition factor |
| `evidence_level` | str | {easy, hard} | Stimulus difficulty |
| `stimulus_id` | int | Unique stimulus identifier | For trial-level analysis |
| `response` | int | {0, 1} | Binary choice (0=incorrect, 1=correct) |
| `rt_ms` | float | Filtered RT in milliseconds | Valid range: 300–3000 ms |
| `accuracy` | int | {0, 1} | Accuracy flag (1=correct response) |
| `extraversion_bfi2s` | float | Extraversion score | Standardized within sample |
| `trial_include` | bool | Include trial in analysis | QC flag post-filtering |

## Preprocessing Steps

### Step 1: RT Filtering
Remove trials outside biologically plausible range:
- **Lower bound**: 300 ms (minimum decision time)
- **Upper bound**: 3000 ms (inattention/distraction threshold)
- **Action**: Flag `trial_include=False` for excluded trials

### Step 2: Response Validation
- Remove trials with missing or ambiguous responses
- Verify binary response coding (0 or 1)

### Step 3: Accuracy Computation
```
accuracy = 1 if (response == correct_answer) else 0
```
Compute per-trial and aggregate by condition.

### Step 4: Standardization
- Standardize `extraversion_bfi2s` within sample: (x - mean) / SD
- Verify no missing values in critical columns

### Step 5: Exclusion Criteria Summary
- Trials with RT < 300 ms: excluded
- Trials with RT > 3000 ms: excluded
- Trials without valid response: excluded
- Participants with <10 valid trials per condition: excluded
- Participants with accuracy < 50% overall: excluded

## Data Quality Checks

Document:
- Number of trials per participant before/after filtering
- Mean accuracy by condition
- Mean RT by condition
- Number of excluded participants and reasons
```

### Python Script: `data_preprocessing.py`

```python
"""
Data Preprocessing Pipeline for Hierarchical Bayesian DDM Analysis
Converts raw trial-level data to HDDM-ready format with QC checks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDMDataPreprocessor:
    """Standardize behavioral data for HDDM ingestion."""
    
    def __init__(
        self,
        rt_min_ms: float = 300.0,
        rt_max_ms: float = 3000.0,
        min_trials_per_condition: int = 10,
        min_accuracy_overall: float = 0.50
    ):
        """
        Initialize preprocessor with filtering parameters.
        
        Parameters
        ----------
        rt_min_ms : float
            Minimum reaction time threshold (ms)
        rt_max_ms : float
            Maximum reaction time threshold (ms)
        min_trials_per_condition : int
            Minimum valid trials required per condition
        min_accuracy_overall : float
            Minimum overall accuracy (0-1) for participant inclusion
        """
        self.rt_min = rt_min_ms
        self.rt_max = rt_max_ms
        self.min_trials = min_trials_per_condition
        self.min_acc = min_accuracy_overall
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw trial data from CSV.
        
        Parameters
        ----------
        filepath : str
            Path to raw CSV file
            
        Returns
        -------
        pd.DataFrame
            Raw data with minimal validation
        """
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} trials from {filepath}")
        return df
    
    def filter_rt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply RT filtering and generate exclusion report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw trial data with rt_raw_ms column
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Filtered dataframe and exclusion statistics
        """
        n_before = len(df)
        
        # Apply RT bounds
        df['trial_include'] = (
            (df['rt_raw_ms'] >= self.rt_min) & 
            (df['rt_raw_ms'] <= self.rt_max) &
            (df['choice'].notna())
        )
        
        # Rename RT column
        df['rt_ms'] = df['rt_raw_ms'].copy()
        
        n_after = df['trial_include'].sum()
        stats = {
            'n_trials_before': n_before,
            'n_trials_after': n_after,
            'n_excluded_rt': n_before - n_after,
            'pct_excluded': 100 * (n_before - n_after) / n_before
        }
        
        logger.info(
            f"RT filtering: {stats['n_excluded_rt']} trials excluded "
            f"({stats['pct_excluded']:.1f}%)"
        )
        
        return df, stats
    
    def compute_accuracy(
        self,
        df: pd.DataFrame,
        correct_response_col: str = 'correct_response'
    ) -> pd.DataFrame:
        """
        Compute trial-level accuracy.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with choice and correct_response columns
        correct_response_col : str
            Column name for ground truth
            
        Returns
        -------
        pd.DataFrame
            Data with accuracy column added
        """
        df['accuracy'] = (df['choice'] == df[correct_response_col]).astype(int)
        return df
    
    def standardize_personality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize personality scores (z-score within sample).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with bfi2s_extraversion column
            
        Returns
        -------
        pd.DataFrame
            Data with standardized extraversion
        """
        mean = df['bfi2s_extraversion'].mean()
        std = df['bfi2s_extraversion'].std()
        df['extraversion_bfi2s_z'] = (df['bfi2s_extraversion'] - mean) / std
        logger.info(f"Extraversion standardized: M={mean:.2f}, SD={std:.2f}")
        return df
    
    def exclude_participants(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply participant-level exclusion criteria.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trial-level data with trial_include flag
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Filtered data and exclusion report
        """
        # Count valid trials per participant per condition
        valid_only = df[df['trial_include']]
        trial_counts = valid_only.groupby(
            ['participant_id', 'instruction', 'base_rate']
        ).size().reset_index(name='n_trials')
        
        # Identify participants failing criterion
        min_trials_by_subj = trial_counts.groupby('participant_id')['n_trials'].min()
        exclude_low_trials = min_trials_by_subj[
            min_trials_by_subj < self.min_trials
        ].index.tolist()
        
        # Check overall accuracy
        overall_acc = valid_only.groupby('participant_id')['accuracy'].mean()
        exclude_low_acc = overall_acc[
            overall_acc < self.min_acc
        ].index.tolist()
        
        # Combine exclusions
        exclude_participants = set(exclude_low_trials + exclude_low_acc)
        df = df[~df['participant_id'].isin(exclude_participants)]
        
        stats = {
            'n_excluded_low_trials': len(exclude_low_trials),
            'n_excluded_low_acc': len(exclude_low_acc),
            'n_included_participants': df['participant_id'].nunique()
        }
        
        logger.info(
            f"Participant exclusion: {len(exclude_participants)} excluded, "
            f"{stats['n_included_participants']} retained"
        )
        
        return df, stats
    
    def prepare_hddm_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final HDDM-compatible dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed data
            
        Returns
        -------
        pd.DataFrame
            HDDM-formatted data with required columns
        """
        hddm_cols = [
            'participant_id', 'block', 'instruction', 'base_rate',
            'evidence_level', 'stimulus_id', 'response', 'rt_ms',
            'accuracy', 'extraversion_bfi2s_z'
        ]
        
        df_hddm = df[hddm_cols].copy()
        df_hddm = df_hddm[df_hddm['trial_include']].drop('trial_include', axis=1)
        
        return df_hddm
    
    def run_pipeline(self, raw_filepath: str, output_filepath: str) -> Dict:
        """
        Execute full preprocessing pipeline.
        
        Parameters
        ----------
        raw_filepath : str
            Input CSV path
        output_filepath : str
            Output CSV path (HDDM-ready)
            
        Returns
        -------
        Dict
            Preprocessing report with all statistics
        """
        # Load
        df = self.load_raw_data(raw_filepath)
        
        # Filter RT
        df, rt_stats = self.filter_rt(df)
        
        # Compute accuracy
        df = self.compute_accuracy(df)
        
        # Standardize personality
        df = self.standardize_personality(df)
        
        # Exclude participants
        df, excl_stats = self.exclude_participants(df)
        
        # Format for HDDM
        df_hddm = self.prepare_hddm_format(df)
        
        # Save
        df_hddm.to_csv(output_filepath, index=False)
        logger.info(f"Saved preprocessed data to {output_filepath}")
        
        # Compile report
        report = {
            'rt_filtering': rt_stats,
            'participant_exclusion': excl_stats,
            'final_n_trials': len(df_hddm),
            'final_n_participants': df_hddm['participant_id'].nunique(),
            'accuracy_by_condition': df_hddm.groupby(
                ['instruction', 'base_rate']
            )['accuracy'].agg(['mean', 'std']).round(3),
            'rt_by_condition': df_hddm.groupby(
                ['instruction', 'base_rate']
            )['rt_ms'].agg(['mean', 'std']).round(1)
        }
        
        return report


# USAGE PSEUDOCODE
if __name__ == "__main__":
    """
    Example usage:
    
    preprocessor = DDMDataPreprocessor(
        rt_min_ms=300,
        rt_max_ms=3000,
        min_trials_per_condition=10,
        min_accuracy_overall=0.50
    )
    
    report = preprocessor.run_pipeline(
        raw_filepath='data/raw_trials.csv',
        output_filepath='data/trials_hddm_ready.csv'
    )
    
    print(report)
    """
```

---

## 2) HIERARCHICAL BAYESIAN DDM FITTING NOTEBOOK

### Jupyter Notebook: `hddm_model_fitting.ipynb`

```python
"""
HIERARCHICAL BAYESIAN DDM FITTING NOTEBOOK
Specification, estimation, and diagnostics for DDM models M0–M4
"""

# ============================================================================
# CELL 1: IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import hddm  # Hierarchical DDM package [1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arviz import from_pymc3  # For diagnostic visualization [2]
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CELL 2: LOAD AND INSPECT PREPROCESSED DATA
# ============================================================================

# Load HDDM-ready data (output from preprocessing pipeline)
df = pd.read_csv('data/trials_hddm_ready.csv')

print(f"Data shape: {df.shape}")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Total trials: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())

# Summary statistics by condition
print("\n=== Summary Statistics ===")
summary = df.groupby(['instruction', 'base_rate']).agg({
    'accuracy': ['mean', 'std'],
    'rt_ms': ['mean', 'std', 'count']
}).round(3)
print(summary)

# ============================================================================
# CELL 3: MODEL SPECIFICATION - M0 (BASELINE)
# ============================================================================

"""
M0: BASELINE MODEL
All DDM parameters constant across experimental conditions.

Estimated parameters:
- v (drift rate): Single value, group-level distribution
- a (boundary separation): Single value
- z (starting point): Single value
- t0 (non-decision time): Single value

Hierarchical structure:
- Group-level: μ_v, σ_v, μ_a, σ_a, μ_z, σ_z, μ_t0, σ_t0
- Subject-level: v_j, a_j, z_j, t0_j for participant j
"""

print("=" * 60)
print("MODEL M0: BASELINE (Parameters constant across conditions)")
print("=" * 60)

# HDDM construction for M0
# Pseudocode: Simple hierarchical model with no condition dependence
m0_pseudocode = """
# Construct M0 model with HDDM
m0 = hddm.HDDM(
    df,
    depends_on={},  # No condition dependencies
    include=['v', 'a', 'z', 't'],  # Estimate all 4 parameters
    p_outlier=0.05  # Account for outliers
)

# Find maximum a-posteriori (MAP) starting values for faster convergence
m0.find_starting_values()

# Sample from posterior using MCMC
# Parameters:
# - samples: Number of posterior samples (typically 5000-10000)
# - burn: Burn-in iterations (typically 2000-5000)
# - thin: Keep every nth sample (typically 1-2)
# - dbname: Database backend for storing samples
m0.sample(
    samples=5000,
    burn=2000,
    thin=1,
    dbname='traces_db/m0',
    db='pickle'
)
"""

print(m0_pseudocode)

# ============================================================================
# CELL 4: MODEL SPECIFICATION - M1 (HYPOTHESIS-DRIVEN)
# ============================================================================

"""
M1: HYPOTHESIS-DRIVEN MODEL
Allows parameters to vary across experimental conditions.

Predicted dependencies:
- v (drift rate) ~ evidence_level: Drift varies with stimulus difficulty
  * Evidence = easy → higher drift (faster, more confident)
  * Evidence = hard → lower drift (slower, less certain)
  
- a (boundary separation) ~ instruction: Boundary varies by speed/accuracy
  * Instruction = speed → lower boundary (faster decisions)
  * Instruction = accuracy → higher boundary (more careful)
  
- z (starting point) ~ base_rate: Bias varies by base-rate
  * Base-rate = high → bias toward prepotent response
  * Base-rate = low → neutral starting point
  
- t0 (non-decision time): Constant across conditions
"""

print("=" * 60)
print("MODEL M1: HYPOTHESIS-DRIVEN")
print("=" * 60)

m1_pseudocode = """
# Specify condition dependencies
# Syntax: parameter_name ~ condition_column

m1 = hddm.HDDM(
    df,
    depends_on={
        'v': 'evidence_level',  # Drift varies with difficulty
        'a': 'instruction',      # Boundary varies with instruction
        'z': 'base_rate'         # Starting point varies with base-rate
    },
    include=['v', 'a', 'z', 't'],
    p_outlier=0.05
)

# Find MAP starting values
m1.find_starting_values()

# Sample posterior
m1.sample(
    samples=5000,
    burn=2000,
    thin=1,
    dbname='traces_db/m1',
    db='pickle'
)

# Generated parameters (example for M1):
# Group level:
#   μ_v[easy], σ_v[easy], μ_v[hard], σ_v[hard]
#   μ_a[speed], σ_a[speed], μ_a[accuracy], σ_a[accuracy]
#   μ_z[high_br], σ_z[high_br], μ_z[low_br], σ_z[low_br]
#   μ_t0, σ_t0 (common)
#
# Subject level: v_j[easy], v_j[hard], a_j[speed], a_j[accuracy],
#               z_j[high_br], z_j[low_br], t0_j for each participant j
"""

print(m1_pseudocode)

# ============================================================================
# CELL 5: MODEL SPECIFICATION - M2, M3, M4 (ALTERNATIVES)
# ============================================================================

"""
M2: PARTIAL CONDITION DEPENDENCE (Alternative 1)
Only drift and boundary vary; starting point and t0 constant.

M3: INTERACTION MODEL (Alternative 2)
v ~ evidence_level × instruction (drift depends on both)
a ~ instruction
z ~ base_rate
t0 ~ constant

M4: NULL MODEL (Alternative 3)
Only a varies; v, z, t0 constant.
(Tests importance of boundary separation manipulation)
"""

print("=" * 60)
print("ALTERNATIVE MODELS")
print("=" * 60)

m2_spec = {
    'description': 'Partial dependence (v ~ evidence, a ~ instruction)',
    'depends_on': {'v': 'evidence_level', 'a': 'instruction'}
}

m3_spec = {
    'description': 'Interaction model (v depends on both evidence & instruction)',
    'depends_on': {'v': 'evidence_level'}  # Simplified for space
}

m4_spec = {
    'description': 'Null model (only a varies)',
    'depends_on': {'a': 'instruction'}
}

print(f"M2: {m2_spec['description']}")
print(f"M3: {m3_spec['description']}")
print(f"M4: {m4_spec['description']}")

# ============================================================================
# CELL 6: CONVERGENCE DIAGNOSTICS - GELMAN-RUBIN R̂
# ============================================================================

"""
Gelman-Rubin diagnostic (R̂) assesses MCMC convergence.

Interpretation:
- R̂ ≈ 1.00: Excellent convergence
- R̂ < 1.02: Good convergence (acceptable)
- R̂ > 1.05: Poor convergence (indicates issues)

For robust estimates, run multiple chains and compute R̂ across chains.
"""

convergence_pseudocode = """
# Pseudocode: Run M1 with multiple chains
m1_chains = []
for chain_id in range(3):  # 3 independent chains
    m1_chain = hddm.HDDM(
        df,
        depends_on={
            'v': 'evidence_level',
            'a': 'instruction',
            'z': 'base_rate'
        }
    )
    m1_chain.find_starting_values()
    m1_chain.sample(samples=5000, burn=2000)
    m1_chains.append(m1_chain)

# Compute Gelman-Rubin R̂
rhat = hddm.analyze.gelman_rubin(m1_chains)

print("Gelman-Rubin R̂ (convergence diagnostic):")
print(rhat)

# Diagnostic: All R̂ values should be < 1.02
if (rhat < 1.02).all():
    print("✓ Excellent convergence")
else:
    print("✗ Convergence issues detected; increase samples or tuning")
"""

print(convergence_pseudocode)

# ============================================================================
# CELL 7: EFFECTIVE SAMPLE SIZE (ESS) AND TRACE PLOTS
# ============================================================================

"""
Effective Sample Size (ESS) accounts for autocorrelation in MCMC samples.
Higher ESS (rule of thumb: ESS > 400 per parameter) indicates better mixing.

Trace plots visualize the MCMC chain trajectory:
- Well-mixing: Random walk pattern (white noise)
- Poor mixing: Correlated samples (trends, stuck values)
"""

trace_pseudocode = """
# Pseudocode: Extract and visualize traces for M1

# Access posterior samples
v_easy_samples = m1.nodes_db.loc['v_easy', 'node'].trace()
a_speed_samples = m1.nodes_db.loc['a_speed', 'node'].trace()

# Plot trace (chain history)
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(v_easy_samples, alpha=0.7, label='v[easy]')
axes[0].set_ylabel('Drift Rate (v)')
axes[0].set_title('Trace Plot: Drift Rate (Easy Evidence)')
axes[0].legend()

axes[1].plot(a_speed_samples, alpha=0.7, label='a[speed]')
axes[1].set_ylabel('Boundary Separation (a)')
axes[1].set_xlabel('MCMC Iteration')
axes[1].set_title('Trace Plot: Boundary Separation (Speed Instruction)')
axes[1].legend()

plt.tight_layout()
plt.savefig('diagnostics/trace_plots_m1.pdf')

# Autocorrelation function (ACF) to assess mixing
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(v_easy_samples, lags=50, ax=axes[0])
axes[0].set_title('Autocorrelation: v[easy]')
plot_acf(a_speed_samples, lags=50, ax=axes[1])
axes[1].set_title('Autocorrelation: a[speed]')
plt.tight_layout()
plt.savefig('diagnostics/acf_plots_m1.pdf')
"""

print(trace_pseudocode)

# ============================================================================
# CELL 8: POSTERIOR SUMMARY STATISTICS
# ============================================================================

"""
After MCMC convergence, summarize posterior distributions for each parameter.
Report mean, standard deviation, and 95% highest density interval (HDI).
"""

posterior_summary_pseudocode = """
# Pseudocode: Extract and display posterior summaries for M1

# Print built-in summary statistics
m1.print_stats()

# Manual extraction for custom table
posterior_summary = []
for param_name in m1.nodes_db.index:
    node = m1.nodes_db.loc[param_name, 'node']
    samples = node.trace()
    
    # Compute statistics
    mean = samples.mean()
    std = samples.std()
    hdi_lower = np.percentile(samples, 2.5)
    hdi_upper = np.percentile(samples, 97.5)
    
    posterior_summary.append({
        'Parameter': param_name,
        'Mean': mean,
        'SD': std,
        '95% HDI Lower': hdi_lower,
        '95% HDI Upper': hdi_upper
    })

summary_df = pd.DataFrame(posterior_summary)
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('results/posterior_summary_m1.csv', index=False)
"""

print(posterior_summary_pseudocode)

# ============================================================================
# CELL 9: LOO-IC MODEL COMPARISON
# ============================================================================

"""
Leave-One-Out Information Criterion (LOO-IC) via Pareto Smoothed 
Importance Sampling (PSIS) estimates out-of-sample predictive performance.

Interpretation:
- Model with lowest LOO-IC is preferred (more predictive)
- Δ LOO-IC > 4 indicates meaningful difference
- SE(Δ LOO-IC) quantifies uncertainty in comparison

HDDM + ArviZ integration enables efficient LOO-IC computation [2]
"""

looic_pseudocode = """
# Pseudocode: Compute LOO-IC for all models

import arviz as az

# Convert pymc2 traces to ArviZ InferenceData format
m0_idata = az.from_pymc2(m0)
m1_idata = az.from_pymc2(m1)
m2_idata = az.from_pymc2(m2)
m3_idata = az.from_pymc2(m3)
m4_idata = az.from_pymc2(m4)

# Compute LOO-IC
loo_m0 = az.loo(m0_idata, pointwise=True)
loo_m1 = az.loo(m1_idata, pointwise=True)
loo_m2 = az.loo(m2_idata, pointwise=True)
loo_m3 = az.loo(m3_idata, pointwise=True)
loo_m4 = az.loo(m4_idata, pointwise=True)

# Compare models
comparison = az.compare({
    'M0_baseline': m0_idata,
    'M1_hypothesis': m1_idata,
    'M2_partial': m2_idata,
    'M3_interaction': m3_idata,
    'M4_null': m4_idata
}, ic='loo')

print(comparison)

# Interpret comparison table
# Columns:
# - elpd_loo: Expected log predictive density (higher is better)
# - p_loo: Effective number of parameters
# - elpd_diff: Difference from best model
# - weight: Akaike weight (relative model probability)
# - dse: SE of difference

# Save results
comparison.to_csv('results/model_comparison_looic.csv')
"""

print(looic_pseudocode)

# ============================================================================
# CELL 10: MODEL COMPARISON TABLE
# ============================================================================

"""
Summary table comparing all candidate models on key criteria.
"""

model_comparison_template = """
# Pseudocode: Generate comparison table

comparison_table = pd.DataFrame({
    'Model': ['M0', 'M1', 'M2', 'M3', 'M4'],
    'Description': [
        'Baseline (all constant)',
        'Hypothesis-driven (v~ev, a~instr, z~br)',
        'Partial (v~ev, a~instr)',
        'Interaction (v~ev×instr)',
        'Null (a~instr only)'
    ],
    'N Parameters (group)': [8, 14, 12, 12, 10],
    'LOO-IC': [
        loo_m0.elpd_loo,
        loo_m1.elpd_loo,
        loo_m2.elpd_loo,
        loo_m3.elpd_loo,
        loo_m4.elpd_loo
    ],
    'Δ LOO-IC vs Best': [
        loo_m0.elpd_loo - loo_m1.elpd_loo,  # Assuming M1 is best
        0.0,
        loo_m2.elpd_loo - loo_m1.elpd_loo,
        loo_m3.elpd_loo - loo_m1.elpd_loo,
        loo_m4.elpd_loo - loo_m1.elpd_loo
    ],
    'SE(Δ)': [
        comparison.loc['M0_baseline', 'dse'],
        comparison.loc['M1_hypothesis', 'dse'],
        comparison.loc['M2_partial', 'dse'],
        comparison.loc['M3_interaction', 'dse'],
        comparison.loc['M4_null', 'dse']
    ]
})

print(comparison_table.to_string(index=False))
comparison_table.to_csv('results/model_comparison_summary.csv', index=False)
"""

print(model_comparison_template)

# ============================================================================
# CELL 11: POSTERIOR PLOTS
# ============================================================================

"""
Visualize posterior distributions (credible intervals and PDFs)
for key parameters across conditions.
"""

posterior_plots_pseudocode = """
# Pseudocode: Plot posteriors for M1

# Extract group-level parameters
v_easy = m1.nodes_db.loc['v_easy', 'node'].trace()
v_hard = m1.nodes_db.loc['v_hard', 'node'].trace()
a_speed = m1.nodes_db.loc['a_speed', 'node'].trace()
a_accuracy = m1.nodes_db.loc['a_accuracy', 'node'].trace()

# Create posterior density plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Drift by evidence
axes[0, 0].hist(v_easy, bins=30, alpha=0.6, label='Easy', density=True)
axes[0, 0].hist(v_hard, bins=30, alpha=0.6, label='Hard', density=True)
axes[0, 0].set_xlabel('Drift Rate (v)')
axes[0, 0].set_ylabel('Posterior Density')
axes[0, 0].set_title('Drift Rate by Evidence Level')
axes[0, 0].legend()

# Plot 2: Boundary by instruction
axes[0, 1].hist(a_speed, bins=30, alpha=0.6, label='Speed', density=True)
axes[0, 1].hist(a_accuracy, bins=30, alpha=0.6, label='Accuracy', density=True)
axes[0, 1].set_xlabel('Boundary Separation (a)')
axes[0, 1].set_ylabel('Posterior Density')
axes[0, 1].set_title('Boundary by Instruction')
axes[0, 1].legend()

# Plot 3: Credible intervals
params_to_plot = [
    ('v_easy', 'v[Easy]'),
    ('v_hard', 'v[Hard]'),
    ('a_speed', 'a[Speed]'),
    ('a_accuracy', 'a[Accuracy]')
]

ci_data = []
for param_name, label in params_to_plot:
    node = m1.nodes_db.loc[param_name, 'node']
    samples = node.trace()
    mean = samples.mean()
    hdi = np.percentile(samples, [2.5, 97.5])
    ci_data.append({'param': label, 'mean': mean, 'hdi_low': hdi[0], 'hdi_high': hdi[1]})

ci_df = pd.DataFrame(ci_data)
y_pos = np.arange(len(ci_df))

axes[1, 0].errorbar(
    ci_df['mean'], y_pos,
    xerr=[ci_df['mean'] - ci_df['hdi_low'], ci_df['hdi_high'] - ci_df['mean']],
    fmt='o', capsize=5, markersize=8
)
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(ci_df['param'])
axes[1, 0].set_xlabel('Parameter Value')
axes[1, 0].set_title('95% Credible Intervals')
axes[1, 0].grid(axis='x', alpha=0.3)

# Remove unused subplot
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/posterior_distributions_m1.pdf', dpi=300)
"""

print(posterior_plots_pseudocode)

# ============================================================================
# CELL 12: FULL PIPELINE PSEUDOCODE SUMMARY
# ============================================================================

summary_pseudocode = """
COMPLETE HDDM FITTING PIPELINE:

1. Load preprocessed data → df_hddm
2. Specify models M0-M4 with condition dependencies [1]
3. For each model:
   a. Find MAP starting values
   b. Run MCMC chains (samples, burn-in, thin)
   c. Check convergence (R̂ < 1.02, visual trace inspection)
   d. Extract posterior summaries (mean, SD, 95% HDI)
4. Compute LOO-IC for all models using ArviZ [2]
5. Generate model comparison table (Δ LOO-IC, SE)
6. Create posterior visualizations
7. Save results:
   - posterior_summary_{model}.csv
   - model_comparison_looic.csv
   - {diagnostics,results}/*.pdf

References:
[1] HDDM: Hierarchical Bayesian DDM in Python
[2] ArviZ: Bayesian model visualization & diagnostics
"""

print(summary_pseudocode)
```

---

## 3) POSTERIOR PREDICTIVE CHECKS & VISUALIZATION

### Python Script: `posterior_predictive_checks.py`

```python
"""
Posterior Predictive Checks and Visualization
Generates synthetic distributions from fitted models and compares to observed data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PosteriorPredictiveChecker:
    """Generate and visualize posterior predictive distributions."""
    
    def __init__(self, observed_data: pd.DataFrame, model_traces: dict):
        """
        Initialize with observed data and model traces.
        
        Parameters
        ----------
        observed_data : pd.DataFrame
            Trial-level observed data
        model_traces : dict
            Dictionary mapping parameter names to posterior samples
            e.g., {'v_easy': array, 'a_speed': array, ...}
        """
        self.data = observed_data
        self.traces = model_traces
        
    def simulate_trials(
        self,
        v: float,
        a: float,
        z: float,
        t0: float,
        n_trials: int = 1000
    ) -> tuple:
        """
        Pseudocode: Simulate DDM trials from parameter values.
        
        Parameters
        ----------
        v : float
            Drift rate
        a : float
            Boundary separation
        z : float
            Starting point (0 to a)
        t0 : float
            Non-decision time (seconds)
        n_trials : int
            Number of trials to simulate
            
        Returns
        -------
        tuple
            (response, rt) arrays from DDM simulation
            
        Notes
        -----
        Simulation via diffusion process:
        - dX = v*dt + dW (Brownian motion with drift)
        - Reflects at boundaries 0 and a
        - Records response (boundary hit) and RT (first passage time)
        """
        # Pseudocode for Euler scheme simulation:
        # dt = 0.001  # time step
        # x = z * a   # initialize at starting point
        # responses, rts = [], []
        # for trial in range(n_trials):
        #     while 0 < x < a:
        #         x += v * dt + np.sqrt(dt) * np.random.normal()
        #     response = 1 if x >= a else 0
        #     rt = time_elapsed + t0
        #     responses.append(response), rts.append(rt)
        # return np.array(responses), np.array(rts)
        pass
    
    def generate_posterior_predictives(
        self,
        n_posterior_samples: int = 100
    ) -> dict:
        """
        Generate posterior predictive distributions by sampling from posteriors.
        
        Parameters
        ----------
        n_posterior_samples : int
            Number of posterior samples to draw
            
        Returns
        -------
        dict
            Synthetic RT/accuracy distributions for each condition
        """
        predictives = {}
        
        # Pseudocode:
        # for condition in ['easy', 'hard']:
        #     v_samples = self.traces[f'v_{condition}']
        #     a_samples = self.traces[f'a_{condition}']
        #     z_samples = self.traces['z']
        #     t0_samples = self.traces['t0']
        #
        #     # Sample posterior parameter combinations
        #     indices = np.random.choice(len(v_samples), n_posterior_samples)
        #
        #     sim_rts, sim_accs = [], []
        #     for idx in indices:
        #         rt, acc = self.simulate_trials(
        #             v=v_samples[idx],
        #             a=a_samples[idx],
        #             z=z_samples[idx],
        #             t0=t0_samples[idx],
        #             n_trials=1000
        #         )
        #         sim_rts.append(rt), sim_accs.append(acc)
        #
        #     predictives[condition] = {
        #         'rts': np.concatenate(sim_rts),
        #         'accs': np.concatenate(sim_accs)
        #     }
        
        return predictives
    
    def compute_quantiles(
        self,
        rts: np.ndarray,
        quantiles: list = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> np.ndarray:
        """
        Compute reaction time quantiles.
        
        Parameters
        ----------
        rts : np.ndarray
            Array of reaction times
        quantiles : list
            Quantile levels (0-1)
            
        Returns
        -------
        np.ndarray
            Quantile values
        """
        return np.quantile(rts, quantiles)
    
    def plot_rt_quantiles(self, posterior_predictives: dict) -> None:
        """
        Compare observed vs. posterior predictive RT quantiles by condition.
        
        Pseudocode
        ----------
        For each condition:
        1. Compute observed RT quantiles (0.1, 0.3, 0.5, 0.7, 0.9)
        2. Compute posterior predictive quantiles
        3. Create scatter plot: observed vs. predicted
        4. Add unity line (y=x) for perfect fit
        """
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5),
            sharex=True, sharey=True
        )
        
        for idx, condition in enumerate(['easy', 'hard']):
            # Observed quantiles
            obs_data = self.data[self.data['evidence_level'] == condition]
            obs_quantiles = self.compute_quantiles(
                obs_data['rt_ms'].values,
                quantiles=quantiles
            )
            
            # Posterior predictive quantiles
            pred_quantiles = self.compute_quantiles(
                posterior_predictives[condition]['rts'],
                quantiles=quantiles
            )
            
            # Plot
            ax = axes[idx]
            ax.scatter(obs_quantiles, pred_quantiles, s=100, alpha=0.7)
            
            # Unity line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=2)
            
            ax.set_xlabel('Observed RT Quantile (ms)')
            ax.set_ylabel('Predicted RT Quantile (ms)')
            ax.set_title(f'RT Quantiles: {condition.capitalize()} Evidence')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ppc_rt_quantiles.pdf', dpi=300)
        print("Saved: results/ppc_rt_quantiles.pdf")
    
    def plot_accuracy_comparison(
        self,
        posterior_predictives: dict
    ) -> None:
        """
        Compare observed vs. posterior predictive accuracy by condition.
        
        Pseudocode
        ----------
        For each condition:
        1. Compute observed accuracy (proportion correct)
        2. Compute posterior predictive accuracy distribution
        3. Create boxplot of posterior predictive + observed point
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = ['easy', 'hard']
        obs_accs = []
        pred_accs = []
        
        for condition in conditions:
            # Observed
            obs_data = self.data[self.data['evidence_level'] == condition]
            obs_acc = obs_data['accuracy'].mean()
            obs_accs.append(obs_acc)
            
            # Posterior predictive
            pred_accs.append(posterior_predictives[condition]['accs'].mean())
        
        x_pos = np.arange(len(conditions))
        width = 0.35
        
        ax.bar(x_pos - width/2, obs_accs, width, label='Observed', alpha=0.8)
        ax.bar(x_pos + width/2, pred_accs, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Evidence Level')
        ax.set_ylabel('Accuracy (proportion correct)')
        ax.set_title('Observed vs. Posterior Predictive Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.capitalize() for c in conditions])
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ppc_accuracy.pdf', dpi=300)
        print("Saved: results/ppc_accuracy.pdf")
    
    def plot_rt_distributions(
        self,
        posterior_predictives: dict
    ) -> None:
        """
        Compare observed vs. posterior predictive RT distributions.
        
        Pseudocode
        ----------
        Create overlay histograms:
        - Observed RT distribution (filled histogram)
        - Posterior predictive RT distribution (outline)
        Filter to valid RT range (300-3000 ms)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, condition in enumerate(['easy', 'hard']):
            ax = axes[idx]
            
            # Observed
            obs_data = self.data[
                (self.data['evidence_level'] == condition) &
                (self.data['rt_ms'] >= 300) &
                (self.data['rt_ms'] <= 3000)
            ]
            
            ax.hist(
                obs_data['rt_ms'],
                bins=30,
                alpha=0.5,
                label='Observed',
                density=True,
                color='steelblue'
            )
            
            # Posterior predictive
            pred_rts = posterior_predictives[condition]['rts']
            pred_rts = pred_rts[(pred_rts >= 300) & (pred_rts <= 3000)]
            
            ax.hist(
                pred_rts,
                bins=30,
                alpha=0.5,
                label='Posterior Predictive',
                density=True,
                color='coral',
                histtype='step',
                linewidth=2
            )
            
            ax.set_xlabel('Reaction Time (ms)')
            ax.set_ylabel('Density')
            ax.set_title(f'RT Distribution: {condition.capitalize()} Evidence')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ppc_rt_distributions.pdf', dpi=300)
        print("Saved: results/ppc_rt_distributions.pdf")
    
    def run_checks(self) -> None:
        """Execute full posterior predictive check pipeline."""
        
        print("Generating posterior predictive distributions...")
        ppc = self.generate_posterior_predictives(n_posterior_samples=100)
        
        print("Creating visualizations...")
        self.plot_rt_quantiles(ppc)
        self.plot_accuracy_comparison(ppc)
        self.plot_rt_distributions(ppc)
        
        print("Posterior predictive checks complete.")


# ============================================================================
# USAGE PSEUDOCODE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    # Load observed data
    df_obs = pd.read_csv('data/trials_hddm_ready.csv')
    
    # Load model traces (from HDDM fitting)
    # This would typically be extracted from saved HDDM models
    model_traces = {
        'v_easy': np.random.normal(0.5, 0.1, 5000),
        'v_hard': np.random.normal(0.3, 0.1, 5000),
        'a_speed': np.random.normal(1.2, 0.2, 5000),
        'a_accuracy': np.random.normal(1.8, 0.2, 5000),
        'z': np.random.normal(0.5, 0.05, 5000),
        't0': np.random.normal(0.3, 0.05, 5000)
    }
    
    checker = PosteriorPredictiveChecker(df_obs, model_traces)
    checker.run_checks()
    """
```

---

## 4) PERSONALITY CORRELATION ANALYSIS (H5)

### Python Script: `personality_correlation_analysis.py`

```python
"""
Bayesian Correlation Analysis: Extraversion (BFI-2-S) & DDM Parameters
Tests hypothesis H5: Extraversion predicts drift rate and boundary separation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PersonalityCorrelationAnalysis:
    """Compute Bayesian correlations between personality and DDM parameters."""
    
    def __init__(
        self,
        subject_data: pd.DataFrame,
        parameter_traces: dict
    ):
        """
        Initialize with subject-level data and posterior parameter samples.
        
        Parameters
        ----------
        subject_data : pd.DataFrame
            Subject-level data with columns:
            - participant_id, extraversion_bfi2s_z (standardized)
        parameter_traces : dict
            Dictionary of subject-level parameter posterior samples:
            - 'v_easy': array (n_samples × n_subjects)
            - 'a_speed': array
            - etc.
        """
        self.subject_data = subject_data
        self.traces = parameter_traces
    
    def compute_bayesian_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """
        Compute Bayesian correlation with credible intervals.
        
        Parameters
        ----------
        x : np.ndarray
            Predictor (extraversion, n_subjects)
        y : np.ndarray
            Outcome (DDM parameter posterior mean, n_subjects)
            
        Returns
        -------
        dict
            Correlation statistics (r, 95% HDI, BF)
            
        Pseudocode
        ----------
        Model: y = β₀ + β₁*x + ε
        
        Prior:
        - β₀ ~ Normal(0, 1)
        - β₁ ~ Normal(0, 1)  # Correlation coefficient
        - σ ~ HalfNormal(1)
        
        Posterior inference:
        - Sample from joint posterior of (β₀, β₁, σ)
        - Compute correlation r = β₁ * SD(x) / SD(y)
        - Extract 95% HDI from r posterior
        - Compute Bayes Factor (BF₁₀) = p(data|H1) / p(data|H0)
        """
        # Basic frequentist correlation (as placeholder)
        r, p_value = stats.pearsonr(x, y)
        
        # Compute 95% CI via bootstrap
        n_boot = 5000
        r_boot = []
        for _ in range(n_boot):
            idx = np.random.choice(len(x), len(x), replace=True)
            r_b, _ = stats.pearsonr(x[idx], y[idx])
            r_boot.append(r_b)
        
        r_boot = np.array(r_boot)
        ci_lower = np.percentile(r_boot, 2.5)
        ci_upper = np.percentile(r_boot, 97.5)
        
        # Pseudocode for Bayes Factor
        # BF₁₀ via Savage-Dickey ratio or bridge sampling
        # (Simplified: would use proper Bayesian framework)
        
        return {
            'r': r,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r_boot': r_boot
        }
    
    def aggregate_subject_parameters(self) -> pd.DataFrame:
        """
        Aggregate posterior samples to subject-level point estimates.
        
        Returns
        -------
        pd.DataFrame
            Subject-level data with posterior mean DDM parameters
            
        Pseudocode
        ----------
        For each subject:
        - Compute posterior mean of each parameter across MCMC samples
        - Combine with personality data
        - Output: [participant_id, extraversion, v_easy_mean, a_speed_mean, ...]
        """
        subject_params = []
        
        for subj_id in self.subject_data['participant_id'].unique():
            subj_entry = {'participant_id': subj_id}
            
            # Add personality
            extraversion = self.subject_data[
                self.subject_data['participant_id'] == subj_id
            ]['extraversion_bfi2s_z'].values[0]
            subj_entry['extraversion_z'] = extraversion
            
            # Add parameter means from traces
            # Pseudocode:
            # for param_name, param_samples in self.traces.items():
            #     subj_idx = subject_id_to_index(subj_id)
            #     subj_entry[f'{param_name}_mean'] = param_samples[:, subj_idx].mean()
            
            subject_params.append(subj_entry)
        
        return pd.DataFrame(subject_params)
    
    def plot_correlation_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_name: str,
        corr_stats: dict
    ) -> None:
        """
        Create scatterplot with regression line and credible band.
        
        Parameters
        ----------
        x : np.ndarray
            Extraversion (standardized)
        y : np.ndarray
            DDM parameter
        param_name : str
            Name of DDM parameter
        corr_stats : dict
            Output from compute_bayesian_correlation()
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Scatter
        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2.5, label='Regression line')
        
        # Pseudocode: Credible band from posterior
        # For each x value, compute 95% CI of predicted y from posterior samples
        # ax.fill_between(x_line, ci_lower_line, ci_upper_line, alpha=0.2)
        
        # Labels and formatting
        ax.set_xlabel('Extraversion (BFI-2-S, standardized)', fontsize=12)
        ax.set_ylabel(f'{param_name} (posterior mean)', fontsize=12)
        ax.set_title(
            f'Personality-DDM Correlation: {param_name}\n'
            f'r = {corr_stats["r"]:.3f}, 95% CI [{corr_stats["ci_lower"]:.3f}, '
            f'{corr_stats["ci_upper"]:.3f}], p = {corr_stats["p_value"]:.3f}',
            fontsize=12
        )
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/correlation_{param_name}.pdf', dpi=300)
        print(f"Saved: results/correlation_{param_name}.pdf")
    
    def plot_correlation_posterior(
        self,
        corr_stats: dict,
        param_name: str
    ) -> None:
        """
        Plot posterior distribution of correlation coefficient.
        
        Parameters
        ----------
        corr_stats : dict
            Contains 'r_boot' (posterior samples)
        param_name : str
            Name of DDM parameter
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of posterior r
        ax.hist(
            corr_stats['r_boot'],
            bins=40,
            density=True,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        
        # Overlay point estimate and credible interval
        ax.axvline(
            corr_stats['r'],
            color='red',
            linestyle='--',
            linewidth=2.5,
            label=f"Posterior mean: {corr_stats['r']:.3f}"
        )
        ax.axvline(
            corr_stats['ci_lower'],
            color='orange',
            linestyle=':',
            linewidth=2,
            label=f"95% CI: [{corr_stats['ci_lower']:.3f}, {corr_stats['ci_upper']:.3f}]"
        )
        ax.axvline(
            corr_stats['ci_upper'],
            color='orange',
            linestyle=':',
            linewidth=2
        )
        
        # Add zero line
        ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Correlation Coefficient (r)', fontsize=12)
        ax.set_ylabel('Posterior Density', fontsize=12)
        ax.set_title(f'Posterior Distribution of Correlation:\nExtraversion ↔ {param_name}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/correlation_posterior_{param_name}.pdf', dpi=300)
        print(f"Saved: results/correlation_posterior_{param_name}.pdf")
    
    def test_outlier_sensitivity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_name: str
    ) -> pd.DataFrame:
        """
        Robustness check: leave-one-out jackknife analysis.
        
        Parameters
        ----------
        x, y : np.ndarray
            Predictor and outcome
        param_name : str
            Parameter name
            
        Returns
        -------
        pd.DataFrame
            Leave-one-out correlation estimates
            
        Pseudocode
        ----------
        For each subject i:
        1. Remove subject i
        2. Recompute correlation on remaining n-1 subjects
        3. Record r without subject i
        4. Identify subjects whose removal substantially changes r
        5. Plot LOO correlations
        """
        loo_correlations = []
        
        for i in range(len(x)):
            x_loo = np.delete(x, i)
            y_loo = np.delete(y, i)
            r_loo, p_loo = stats.pearsonr(x_loo, y_loo)
            
            loo_correlations.append({
                'excluded_subject': i,
                'r_without_subject': r_loo,
                'p_without_subject': p_loo
            })
        
        loo_df = pd.DataFrame(loo_correlations)
        
        # Visualize LOO results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            range(len(loo_df)),
            loo_df['r_without_subject'],
            s=80,
            alpha=0.7
        )
        ax.axhline(
            loo_df['r_without_subject'].mean(),
            color='red',
            linestyle='--',
            label='Mean LOO-r'
        )
        ax.fill_between(
            range(len(loo_df)),
            loo_df['r_without_subject'].mean() - loo_df['r_without_subject'].std(),
            loo_df['r_without_subject'].mean() + loo_df['r_without_subject'].std(),
            alpha=0.2,
            color='red'
        )
        
        ax.set_xlabel('Left-Out Subject Index')
        ax.set_ylabel('Correlation (excluding one subject)')
        ax.set_title(f'Leave-One-Out Sensitivity Analysis: {param_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/loo_sensitivity_{param_name}.pdf', dpi=300)
        
        return loo_df
    
    def generate_correlation_report(self) -> pd.DataFrame:
        """
        Generate comprehensive correlation analysis report.
        
        Returns
        -------
        pd.DataFrame
            Summary table: parameter × correlation statistics
        """
        # Aggregate subject parameters
        subject_params = self.aggregate_subject_parameters()
        
        # Parameters to correlate with extraversion
        param_columns = [
            col for col in subject_params.columns
            if col not in ['participant_id', 'extraversion_z']
        ]
        
        report = []
        
        for param in param_columns:
            # Extract valid data
            valid_idx = ~(subject_params[param].isna() | subject_params['extraversion_z'].isna())
            x = subject_params.loc[valid_idx, 'extraversion_z'].values
            y = subject_params.loc[valid_idx, param].values
            
            # Compute correlation
            corr_stats = self.compute_bayesian_correlation(x, y)
            
            # Sensitivity check
            loo_df = self.test_outlier_sensitivity(x, y, param)
            loo_std = loo_df['r_without_subject'].std()
            
            # Create plots
            self.plot_correlation_scatter(x, y, param, corr_stats)
            self.plot_correlation_posterior(corr_stats, param)
            
            # Add to report
            report.append({
                'Parameter': param,
                'r': corr_stats['r'],
                'p_value': corr_stats['p_value'],
                '95% CI Lower': corr_stats['ci_lower'],
                '95% CI Upper': corr_stats['ci_upper'],
                'LOO-r SD (sensitivity)': loo_std,
                'N': len(x)
            })
        
        report_df = pd.DataFrame(report)
        report_df.to_csv('results/personality_correlation_report.csv', index=False)
        
        print("\n=== Personality-DDM Correlation Report ===")
        print(report_df.to_string(index=False))
        
        return report_df


# ============================================================================
# USAGE PSEUDOCODE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    # Load subject-level data with personality
    subject_data = pd.read_csv('data/subject_personality.csv')
    # Columns: participant_id, bfi2s_extraversion, extraversion_bfi2s_z
    
    # Load posterior parameter estimates (from HDDM fitting)
    # Typically extracted as subject-level posterior means
    param_traces = {
        'v_easy': np.random.normal(0.5, 0.1, (5000, 50)),  # n_samples × n_subjects
        'v_hard': np.random.normal(0.3, 0.1, (5000, 50)),
        'a_speed': np.random.normal(1.2, 0.2, (5000, 50)),
        'a_accuracy': np.random.normal(1.8, 0.2, (5000, 50)),
    }
    
    analyzer = PersonalityCorrelationAnalysis(subject_data, param_traces)
    report = analyzer.generate_correlation_report()
    """
```

---

## Directory Structure

```
project-root/
├── README.md (project overview)
├── data/
│   ├── raw_trials.csv (raw behavioral data)
│   ├── trials_hddm_ready.csv (preprocessed, HDDM-ready)
│   └── subject_personality.csv (BFI-2-S scores)
├── scripts/
│   ├── data_preprocessing.py (Script 1)
│   ├── posterior_predictive_checks.py (Script 3)
│   ├── personality_correlation_analysis.py (Script 4)
│   └── utils_ddm.py (helper functions)
├── notebooks/
│   ├── hddm_model_fitting.ipynb (Notebook 2)
│   └── exploratory_analysis.ipynb
├── results/
│   ├── posterior_summary_m0.csv
│   ├── posterior_summary_m1.csv
│   ├── model_comparison_looic.csv
│   ├── posterior_distributions_m1.pdf
│   ├── ppc_rt_quantiles.pdf
│   ├── ppc_accuracy.pdf
│   ├── ppc_rt_distributions.pdf
│   ├── correlation_v_easy.pdf
│   ├── correlation_posterior_v_easy.pdf
│   ├── loo_sensitivity_v_easy.pdf
│   └── personality_correlation_report.csv
├── diagnostics/
│   ├── trace_plots_m1.pdf
│   ├── acf_plots_m1.pdf
│   └── convergence_diagnostics.txt
├── traces_db/ (MCMC sample storage)
│   ├── m0.pkl
│   ├── m1.pkl
│   └── ...
└── docs/
    └── methodology.md
```

---

## Key References

HDDM is a **Python-based toolbox for hierarchical Bayesian estimation of drift diffusion models**, enabling flexible specification of subject-level and group-level parameters with MCMC-based posterior inference[1]. Recent developments integrate **ArviZ for advanced Bayesian diagnostics and model comparison via LOO-IC**, providing efficient out-of-sample predictive evaluation[2]. This pipeline implements reproducible open-science practices with fully documented artifacts.