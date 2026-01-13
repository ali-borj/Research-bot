# Research: The Social Brain

**Date:** 2026-01-13 18:15

**AI Pipeline:** Perplexity (sonar-pro + sonar-reasoning-pro) → Gemini 2.5

---

# Face Pareidolia and Perceptual Decision-Making: A Hierarchical Bayesian Drift Diffusion Modeling Study

---

## Title Page

**Face Pareidolia and Perceptual Decision-Making: A Hierarchical Bayesian Drift Diffusion Modeling Study of Evidence Strength, Speed–Accuracy Trade-Offs, and Base-Rate Biases**

**Proposed Laboratory Experiment**

**A Publication-Ready Research Protocol**

*Prepared: January 2026*

---

## Abstract

Face pareidolia—the illusory perception of faces in ambiguous or non-face objects—provides a window into how the visual and decision-making systems interact under perceptual uncertainty[1][2]. Despite growing behavioral and neural evidence that pareidolia reflects rapid, early face-processing mechanisms[2], the latent cognitive processes governing "face" vs. "non-face" decisions remain incompletely understood. This preregistered study applies hierarchical Bayesian drift diffusion modeling (HDDM) to decompose reaction times and accuracy in a 2×2 within-subject design manipulating (A) decision instructions (Speed vs. Accuracy) and (B) face base-rate (High [70%] vs. Low [30%] face-present trials). Stimuli span three evidence levels: real human faces (high evidence), pareidolic images from the "Faces in Things" dataset[1] (ambiguous evidence), and matched non-face objects (low evidence). We predict that real faces will yield higher drift rates (*v*) than pareidolia or non-faces, that accuracy instructions will increase boundary separation (*a*), that high face base-rate will increase starting-point bias (*z*) toward the "face" boundary, and that pareidolia will show intermediate parameter values reflecting their ambiguous perceptual status. *N* = 35 healthy adults; session duration ~20 minutes. Primary analysis uses parsimonious HDDM (estimating *v*, *a*, *z*, *t*₀ per condition) with posterior predictive checks and model comparison via leave-one-out information criterion. This study demonstrates how a validated cognitive modeling framework can isolate the mechanistic origins of face perception illusions, with implications for understanding normal vision, individual differences, and clinical manifestations of elevated pareidolia (e.g., schizophrenia)[4].

---

## Introduction

### Background: Pareidolia as a Window into Face Processing

The human visual system exhibits a remarkable and sometimes costly bias toward detecting faces in the environment. This bias occasionally produces false positives: observers report seeing faces in clouds, in wood grain, in product designs, and in randomized patterns—a phenomenon termed **pareidolia**[1][2]. Rather than representing a failure of perception, pareidolia likely reflects adaptive overapproximation by neural circuits specialized for rapid face detection. (Citation needed: A specific neuroscience source directly contrasting adaptive vs. error-based accounts of pareidolia. Expected evidence: theoretical paper or review contrasting evolutionary costs of missed vs. false-positive face detections.) Recent neuroimaging studies have shown that pareidolic faces activate the same face-selective regions (fusiform face area, occipital face area) as veridical faces, and do so with similar latency (~160 ms)[2], suggesting that the illusion emerges early in visual processing rather than through late, post-perceptual reinterpretation.

However, detecting a face in a stimulus and *deciding* whether to report it as a face are distinct operations. Reaction times and accuracy in pareidolia tasks are influenced not only by the visual stimulus but also by task instructions, expectations, and individual differences. Current evidence does not specify how these factors interact at the level of evidence accumulation and decision commitment—the core mechanisms captured by drift diffusion modeling.

### Theoretical Rationale: Drift Diffusion Modeling of Pareidolia

The **drift diffusion model (DDM)** is a well-validated framework for decomposing binary perceptual and cognitive decisions into latent cognitive parameters[citation needed: foundational DDM paper such as Ratcliff & McKoon, 2008. Expected evidence: standard reference in decision neuroscience establishing DDM as the model of choice for two-choice reaction time and accuracy]. The standard DDM assumes that a decision variable accumulates stochastic evidence over time, starting at a bias point (*z*) and terminating when it reaches either an upper or lower boundary (separated by distance *a*). The rate of accumulation toward the correct boundary is the **drift rate** (*v*), which is proportional to stimulus quality or evidence strength. Non-decision processes (motor execution, stimulus encoding) are captured by *t*₀.

For face pareidolia, this decomposition yields interpretable predictions:

1. **Evidence strength** (*v*): Real faces should produce faster, more reliable evidence accumulation than pareidolic images, which in turn should accumulate faster than non-face objects. This is supported by the existence of a "Goldilocks zone" for pareidolia[1]—stimuli of intermediate complexity evoke peak pareidolic responses, consistent with ambiguous or weak evidence.

2. **Decision caution** (*a*): Instructions to prioritize accuracy over speed typically increase boundary separation[citation needed: empirical DDM study showing speed-accuracy trade-off effects on boundary separation. Expected evidence: paper showing that accuracy instructions increase *a* and slow RT while improving accuracy, particularly under perceptual uncertainty]. In pareidolia tasks, accuracy-favoring instructions might encourage stronger evidence accumulation before committing to a "face" decision, especially for ambiguous stimuli.

3. **Starting-point bias** (*z*): Task context shapes prior expectations. A high base-rate of face-present trials (70% face) should bias the starting point closer to the "face" boundary, reducing the evidence needed to reach that boundary, consistent with Bayesian updating of face expectancy[citation needed: DDM study of base-rate manipulation on starting-point bias. Expected evidence: paper showing that high base-rate of target-present trials biases *z* toward the target response boundary]. Conversely, a low base-rate (30% face) should bias *z* toward "non-face."

4. **Non-decision time** (*t*₀): Pareidolic and non-face stimuli may require additional stimulus encoding or feature verification[2], potentially increasing *t*₀.

Together, these parameters provide a mechanistic explanation for behavioral patterns that raw RT and accuracy measures alone cannot isolate.

### Gap in the Literature

While pareidolia research has flourished in neuroscience and psychology[1][2][3], producing datasets, neural models, and behavioral characterizations, *no study has yet applied hierarchical Bayesian DDM to decompose pareidolia decisions in a design that systematically manipulates evidence strength, task instructions, and base-rate expectations*. This gap is significant because:

- It prevents direct testing of whether pareidolia effects arise from changes in evidence strength (*v*) vs. decision bias (*z*).
- It leaves unspecified how task-level manipulations (speed-accuracy trade-off, base-rate cuing) interact with perceptual ambiguity in driving pareidolia illusions.
- It obscures potential individual differences in decision mechanisms (e.g., does elevated pareidolia in schizophrenia reflect weaker evidence or stronger bias?)[4].

---

## State of the Art

### Behavioral and Neural Characterization of Pareidolia

Recent work has established pareidolia as a rapid, automatic process rooted in early visual responses. Ward and colleagues (2020) used fMRI and MEG in tandem, presenting human participants with three stimulus classes: real human faces, pareidolic images (objects with face-like features), and matched control objects without illusory faces[2]. Brain imaging revealed that pareidolic faces recruited face-selective regions—fusiform face area (FFA) and occipital face area (OFA)—with response latencies (~160 ms) comparable to those for genuine faces, and faster than would be expected if pareidolia depended on slower, cognitive reinterpretation[2]. This finding aligns with a bottom-up, feature-driven account of face perception.

Complementary work by Ward and Smith (2024) introduced a rich dataset, "Faces in Things," comprising 5,000 web images with human-annotated pareidolic faces, alongside mathematical models predicting when pareidolia should occur[1]. Their key insight is the **"Goldilocks zone"**: pareidolia is maximal at intermediate levels of stimulus complexity (filter width ~16 in their Fourier-based noise stimuli). This inverted-U relationship suggests that pareidolia arises when stimuli contain enough face-like structure to activate face detectors but retain sufficient ambiguity to permit illusory interpretation. Critically, their behavioral experiments confirmed that human observers show peak pareidolia detection rates at the model-predicted noise complexity, and this pattern generalizes across subjects[1].

Computational models corroborate these findings. Deep convolutional neural networks (CNNs) exhibit human-like pareidolia via representational similarity to brain data[citation needed: full source is noted as Güçlü et al. (2024) in SOURCE_BUNDLE but details incomplete. Expected evidence: paper showing CNN representations of pareidolic and real faces are highly correlated via representational dissimilarity matrices and neural encoding models]. Clinical evidence further supports pareidolia as a measurable perceptual phenomenon: patients with schizophrenia show elevated rates of face pareidolia compared to healthy controls and bipolar patients, with the enhanced pareidolia correlating with psychotic symptom severity[4]. This suggests a tractable neurobiological variation in face processing that could be indexed by DDM parameters.

### Drift Diffusion Modeling in Perceptual Decision Tasks

The DDM framework has proven invaluable for decomposing binary perceptual decisions across domains (visual motion, letter identification, lexical judgment). Avery et al. (2023) extended the standard DDM to capture **endogenous effects**—ways in which internal states (e.g., fatigue, attention) non-linearly modulate evidence strength—in a non-linear DDM (nl-DDM) applied to decision data[citation needed: full Avery et al. (2023) reference and explanation of how nl-DDM improves identifiability of v/a/z. Expected evidence: methodological paper demonstrating that standard DDM parameter correlations are reduced in nl-DDM under specific conditions]. Such extensions highlight the importance of careful model specification to avoid confounding drift rate changes with boundary or bias shifts.

The application of DDM to face-processing tasks remains sparse, though related work on attentional biases in face perception hints at the utility of this approach. (Citation needed: A published DDM study of base-rate or expectancy effects in face detection/recognition. Expected evidence: paper using DDM to quantify how prior expectations (e.g., self-face likelihood, familiar face base-rate) modulate *v*, *z*, or *a* in a face task.) Such a study would directly support our hypothesis that base-rate manipulations affect *z* in pareidolia judgments.

### Hierarchical Bayesian Fitting and Model Comparison

Hierarchical Bayesian DDM (HDDM) represents a major methodological advance, allowing simultaneous estimation of group-level and individual-level parameters while accounting for within-subject correlations and shrinkage[citation needed: HDDM foundational paper such as Wiecki et al. (2013). Expected evidence: computational modeling paper introducing HDDM and demonstrating improved parameter recovery vs. maximum-likelihood fitting, especially at small N]. HDDM is particularly suited to the present study because it enables pooling of information across participants and conditions (e.g., learning from multiple pareidolia trials to constrain the pareidolia drift rate *v*) while respecting hierarchical structure in the data. Model comparison via leave-one-out information criterion (LOO-IC) permits quantitative assessment of whether adding parameters (e.g., condition-specific *z*) meaningfully improves predictions compared to simpler alternatives.

---

## Research Questions & Hypotheses

### Primary Research Questions

1. **Do real faces, pareidolic images, and non-face objects elicit distinct drift rates (*v*) in a binary face/non-face decision task?**

2. **Does accuracy-emphasized instruction increase boundary separation (*a*) compared to speed-emphasized instruction, and does this effect interact with stimulus type (real vs. pareidolic vs. non-face)?**

3. **Does high face base-rate (70%) shift starting-point bias (*z*) toward the "face" response boundary compared to low base-rate (30%)?**

4. **Can DDM parameter estimates predict individual differences in overall pareidolia susceptibility (e.g., number of faces "detected" across all conditions)?**

### Formal Hypotheses

| Hypothesis | Manipulation | Predicted Outcome | DDM Parameter | Direction |
|-----------|--------------|------------------|---|---|
| **H1** | Evidence strength (Real > Pareidolia > Non-face) | Real faces accumulate faster; pareidolia intermediate; non-faces slower | *v*_Real > *v*_Pareidolia > *v*_Non-face | Monotonic decrease |
| **H2** | Instruction (Accuracy > Speed) | Accuracy instruction increases decision caution | *a*_Accuracy > *a*_Speed | Increase |
| **H3** | Base-rate (High [70%] > Low [30%] face) | High face base-rate biases accumulation toward "face" boundary | *z*_High-BR > *z*_Low-BR | Shift upward toward "face" |
| **H4** | Accuracy × Evidence interaction | Accuracy instruction increases *a* most for ambiguous (pareidolic) stimuli | Interaction: *a*_Accuracy,Pareidolia > *a*_Speed,Pareidolia | Evidence × Instruction |
| **H5** | Individual differences in pareidolia susceptibility | Higher *v* and *z* bias in pareidolia condition predicts more "face" responses | Correlation: *v*_Pareidolia, *z*_High-BR with face accuracy | Positive correlation |

---

## Methods

### Participants

**Target N**: 35 healthy adults (18–35 years, no known neurological or psychiatric disorders). **Justification for N**: At N=35, power to detect a within-subject contrast of *d* ≈ 0.6 (e.g., *v*_Real vs. *v*_Pareidolia) with ≥80% power in a paired *t*-test (two-tailed, α = 0.05) is achieved. HDDM benefits from larger N, but feasibility constraints (20-minute session, single-lab capacity) and typical effect sizes in perceptual DDM studies[citation needed: meta-analytic or empirical estimate of effect sizes for DDM parameters under stimulus manipulations in perception. Expected evidence: prior empirical paper showing *v* differences of 0.3–1.0 SD units across stimulus quality levels] justify N=35 as adequate for detecting primary effects while remaining feasible.

**Exclusion criteria**: history of psychosis, bipolar disorder, or substance dependence (assessed via screener); uncorrected vision; color blindness.

**Recruitment**: undergraduate psychology participant pool; online consent and questionnaires 24–48 hours before lab session.

### Materials and Stimuli

**Stimulus set composition**:
- **Real faces**: 20 high-quality photographs of neutral-expression faces (diverse age, ethnicity, sex) from standard face databases (to be sourced from open databases; e.g., MORPH, UTK Face).
- **Pareidolic images**: 20 images from the "Faces in Things" dataset[1], selected to span the Goldilocks zone (filter width ~14–18, optimal for peak pareidolia[1]).
- **Non-face control objects**: 20 matched images (textures, tools, animals, or scenes without obvious face-like structure) selected to be visually distinct from pareidolic images but similar in luminance and contrast.

**Stimulus presentation**: 400 × 400 pixel images, centered on a gray background, displayed for 500 ms followed by a blank screen until response.

**Justification**: The three-level stimulus hierarchy (real > pareidolia > non-face) directly tests the hypothesis that evidence strength (parameterized by *v*) scales with face-likeness. Pareidolia stimuli are sourced from a validated dataset with established behavioral properties[1], ensuring ecological validity and reproducibility.

### Experimental Design

**Within-subject factors**:
- **Instruction** (2 levels): Speed-emphasized vs. Accuracy-emphasized (blocked).
- **Base-rate** (2 levels): High-face (70% face-present) vs. Low-face (30% face-present) (blocked).
- **Evidence level** (3 levels): Real face, Pareidolic, Non-face (mixed within blocks).

**Design structure**: 2 (Instruction) × 2 (Base-rate) × 3 (Evidence) = 12 condition cells, all within-subject. Trials are grouped into four **blocks** corresponding to the four Instruction × Base-rate combinations:

1. **Block 1: Speed-Accuracy, High-Face base-rate** (Speed instructions, 70% face trials)
2. **Block 2: Speed-Accuracy, Low-Face base-rate** (Speed instructions, 30% face trials)
3. **Block 3: Accuracy-Emphasized, High-Face base-rate** (Accuracy instructions, 70% face trials)
4. **Block 4: Accuracy-Emphasized, Low-Face base-rate** (Accuracy instructions, 30% face trials)

**Block order counterbalancing**: Participants are pseudo-randomly assigned to one of four orderings (Latin square counterbalancing for main factors):
- Order A: Block 1 → 2 → 3 → 4
- Order B: Block 1 → 3 → 2 → 4
- Order C: Block 3 → 1 → 4 → 2
- Order D: Block 3 → 4 → 1 → 2

This ensures that Instruction and Base-rate effects are not confounded with block position.

### Trial Sequence and Timing

**Session structure (≈20 minutes total)**:

| Segment | Duration | Content |
|---------|----------|---------|
| Instructions & practice | 3 min | Written instructions for Speed block; 10 practice trials (varied evidence levels) |
| Block 1 (Speed, High-BR) | 4 min | 45 trials (31 face, 14 non-face per 70/30 ratio); includes 2×minute breaks |
| Brief transition | 1 min | Instructions for Accuracy block; 3 practice trials |
| Block 2 (Speed, Low-BR) | 4 min | 45 trials (13 face, 32 non-face per 30/70 ratio); includes 1×minute break |
| Mid-session break | 1 min | Offered water, stretching |
| Block 3 (Accuracy, High-BR) | 4 min | 45 trials (31 face, 14 non-face); includes 1×minute break |
| Block 4 (Accuracy, Low-BR) | 4 min | 45 trials (13 face, 32 non-face); includes 1×minute break |
| Debrief & exit | 1 min | Thank you; schedule debriefing (separate session if needed) |

**Total trials**: 180 (4 blocks × 45 trials). **Justification for trial count**: 180 trials across 12 condition cells yields ~15 observations per cell, sufficient for stable HDDM parameter estimation at the individual level[citation needed: prior work on required trial counts per condition in HDDM. Expected evidence: simulation or empirical study showing that 15–20 trials per condition recovers DDM parameters with < 10% relative bias].

**Stimulus allocation within each block**:
- Each block presents 45 trials split across three evidence levels (15 Real, 15 Pareidolia, 15 Non-face).
- Face base-rate is maintained via the proportion of trials containing a face-category stimulus (Real + Pareidolia) vs. non-face (Non-face). For High-BR blocks, 31 of 45 trials are face-category (Real + Pareidolia combined); for Low-BR blocks, 13 of 45 trials are face-category.
- Example: Block 1 (High-BR, 70%) could allocate 15 Real, 16 Pareidolia, 14 Non-face.

**Response window**: No hard timeout. Participants are instructed: "In the **Speed** block, respond as quickly as possible while maintaining good accuracy. In the **Accuracy** block, respond only when you are confident in your judgment; accuracy is more important than speed." This open-ended response window avoids artificial truncation of the RT distribution while naturally inducing speed-accuracy trade-offs via instructions[citation needed: justification for open-ended vs. deadline-based RT collection in DDM experiments. Expected evidence: paper comparing drift diffusion parameter recovery under different response deadline structures, showing minimal bias with open-ended deadlines in perceptual tasks].

### Questionnaires (Pre-Session, Online)

**BFI-2-S (Big Five Inventory-2 Short Form)**: 30-item self-report measuring five personality dimensions. We focus on **Extraversion** (6 items; e.g., "I am outgoing, sociable") as a secondary, exploratory predictor of individual differences in pareidolia susceptibility. (Citation needed: empirical justification for why Extraversion should predict pareidolia. Expected evidence: theory or prior study linking social motivation/approach orientation to face detection bias.) Cronbach's α target ≥ 0.70 for the Extraversion subscale in the present sample.

**Rationale for Extraversion (exploratory)**: At N=35, statistical power to detect a moderate correlation (*r* ≈ 0.40) between Extraversion and a focal DDM parameter (e.g., *v*_Pareidolia) is ~70% (two-tailed, α = 0.05). We treat Extraversion as a secondary, correlational predictor and avoid causal language (e.g., no mediation hypotheses) given the modest sample size and exploratory nature.

---

## Analysis Plan

### Preprocessing

**RT filtering**: Exclude trials with RT < 200 ms (anticipatory) or RT > 5000 ms (inattention). Expect <2% of trials to be removed.

**Accuracy screening**: Identify participants with overall accuracy <55% (near chance) across the entire session; such data may indicate task misunderstanding. Expect <5% of participants. For borderline cases, retain but note in sensitivity analyses.

### Primary DDM Model

**Model specification**: Estimate five parameters per condition (Evidence × Instruction × Base-rate combination):

\[
\text{DDM}(\text{RT, Choice}) = f(v, a, z, t_0, \tau_v)
\]

Where:
- **v** (drift rate): Evidence strength for "face" vs. "non-face" response, allowed to vary by Evidence level (Real, Pareidolia, Non-face).
- **a** (boundary separation): Decision caution, allowed to vary by Instruction (Speed, Accuracy).
- **z** (starting point, as proportion of *a*): Bias toward "face" response, allowed to vary by Base-rate (High-face, Low-face).
- **t₀** (non-decision time): Encoding + motor execution, estimated as a group-level hyperparameter or allowed to vary by Evidence level (real faces might require faster encoding).
- **τ_v** (across-trial variability in drift): Included only if model comparison supports it; otherwise set to zero.

**Hierarchical structure** (HDDM framework):

For participant *i* and condition *j*, the observed RT and choice are generated from a DDM with subject-level parameters:

\[
v_{ij} \sim \mathcal{N}(\mu_v^{(j)}, \sigma_v^{(j)}), \quad a_{ij} \sim \mathcal{N}(\mu_a^{(j)}, \sigma_a^{(j)}), \quad z_{ij} \sim \text{Beta}(\alpha, \beta)
\]

And group-level hyperpriors:

\[
\mu_v^{(j)} \sim \mathcal{N}(0, 2), \quad \sigma_v^{(j)} \sim \text{HalfNormal}(1), \quad \text{etc.}
\]

**Model formula example** (pseudo-code for HDDM syntax):
```
v ~ C(Evidence) + C(Instruction) + C(Evidence):C(Instruction)
a ~ C(Instruction)
z ~ C(Base-rate)
t0 ~ 1  [group-level, fixed]
```

This allows *v* to depend on Evidence, Instruction, and their interaction; *a* to depend on Instruction; *z* to depend on Base-rate; and *t₀* to be a shared group-level constant.

**Justification**: This parsimonious specification tests focal hypotheses (H1–H3) while avoiding overparameterization. We deliberately do NOT include Evidence × Base-rate interactions on *z* or Instruction × Base-rate interactions on *a*, as these are not predicted and would reduce identifiability. (Citation needed: theoretical justification for why base-rate manipulations should not interact with instruction-induced boundary changes. Expected evidence: decision theory paper or prior DDM study supporting parameter independence assumptions.)

### Bayesian Inference and Model Checking

**Sampling**: Use MCMC sampling (e.g., NUTS sampler in PyMC3 or Stan) with ≥2 parallel chains, 2000 warmup iterations, and 2000 post-warmup iterations per chain. Target **R̂ < 1.01** for all parameters and **effective sample size (ESS) > 400** for focal parameters. Include trace plots in supplementary materials.

**Convergence diagnostics**: If R̂ > 1.05 for any focal parameter, refit with increased iterations or prior adjustments. Prepare a fallback analysis using maximum-likelihood DDM fitting (e.g., scipy.optimize) if MCMC fails to converge.

**Posterior summaries**: Report posterior means, 95% highest posterior density (HPD) credible intervals, and point estimates for publication.

### Posterior Predictive Checks (PPC)

Simulate 1000 datasets from posterior predictive distribution and compare to observed data on:

1. **RT quantiles by condition**: Mean, median, and 0.9 quantile RT for each stimulus type and instruction condition.
2. **Accuracy by condition**: Overall proportion "face" responses by Evidence × Instruction × Base-rate.
3. **RT–accuracy relationship**: Conditional accuracy function (CAF), binning trials by RT quartile.

Visualize observed vs. simulated distributions; quantify discrepancy via Kolmogorov–Smirnov test (p > 0.05 indicates adequate fit).

### Model Comparison

Compare three candidate models:

**Model 1 (Focal)**: As specified above (v ∝ Evidence + Instruction; a ∝ Instruction; z ∝ Base-rate).

**Model 2 (Alternative A)**: Constrain z to 0.5 (no base-rate effect), testing whether base-rate influences *v* instead.

**Model 3 (Alternative B)**: Include Evidence × Base-rate interaction on *z*, testing whether pareidolia bias is modulated by base-rate differently than real faces.

Use **leave-one-out information criterion (LOO-IC)** for model comparison. Report elpd_diff and its SE for each pairwise comparison. A difference of >4 elpd points constitutes meaningful evidence for one model over another.

### Individual Differences Analysis

**Correlation with behavior**: Estimate Pearson correlations between subject-level posterior means for *v*_Pareidolia, *z*_High-BR, and:
- Overall "face" response rate across all pareidolia trials (e.g., number of pareidolia trials labeled "face" / total pareidolia trials).
- Pareidolia susceptibility score from the "Faces in Things" dataset[1] (if participants also rate pareidolic images on a face-likeness scale in a post-session questionnaire).

**Exploratory personality analysis**: Correlate Extraversion (BFI-2-S) with *v*_Pareidolia, *a*_Accuracy, and *z*_High-BR. Use Bayesian correlation to avoid multiple-comparisons burden[citation needed: justification for Bayesian correlation in exploratory analyses. Expected evidence: methodological paper on controlling Type I error in exploratory personality-cognition correlations using Bayesian model averaging or default priors].

---

## Expected Results

### Primary Predictions

**Evidence strength (*v*)**: We predict *v*_Real > *v*_Pareidolia > *v*_Non-face, with effect sizes:
- *v*_Real − *v*_Pareidolia ≈ 0.8 (Cohen's *d*), reflecting the clear face identity in real photos.
- *v*_Pareidolia − *v*_Non-face ≈ 0.4, reflecting ambiguity in pareidolic images[1].

These patterns reflect the "Goldilocks zone" finding: pareidolia yields intermediate evidence strength, stronger than non-faces but weaker than real faces[1].

**Boundary separation (*a*)**: We expect *a*_Accuracy > *a*_Speed, with *d* ≈ 0.6. Accuracy instructions increase caution (larger *a*) as well-established in DDM studies (Citation needed: reference for speed-accuracy trade-off on boundary). The effect may be larger for ambiguous (pareidolia) stimuli, reflecting increased hesitation when evidence is weak.

**Starting-point bias (*z*)**: We expect *z*_High-BR > *z*_Low-BR, with *d* ≈ 0.5. High face base-rate (70%) shifts *z* upward (toward "face" response), reducing the evidence needed to respond "face." Conversely, low base-rate (30%) shifts *z* downward.

### Secondary Predictions

**Non-decision time (*t₀*)**: Pareidolia and non-face stimuli may require longer stimulus encoding (e.g., feature ambiguity resolution), yielding *t₀*_Pareidolia, *t₀*_Non-face > *t₀*_Real by 20–50 ms.

**Individual differences**: Subject-level *v*_Pareidolia and *z*_High-BR should correlate positively (*r* ≈ 0.35–0.50) with overall pareidolia susceptibility (number of "face" responses to pareidolia images), supporting the mechanistic account.

**Extraversion (exploratory)**: We tentatively predict that Extraversion correlates positively with *v*_Pareidolia (*r* ≈ 0.25–0.40), reflecting greater approach-motivation or social attention[citation needed: theory linking extraversion to face-detection bias. Expected evidence: social neuroscience paper showing extraversion modulates face processing or social approach motivation]. However, this is exploratory and should be treated as hypothesis-generating.

### Bayesian Model Comparison

Model 1 (focal) is expected to show superior LOO-IC relative to Models 2 and 3, with elpd_diff ≥ 4.

---

## Discussion & Implications

### Interpretation of Findings

If results align with predictions, this study will provide the first mechanistic decomposition of face pareidolia using DDM, demonstrating that:

1. **Pareidolia arises from ambiguous visual evidence (intermediate *v*)**, not from inflexible bias (i.e., *z* does not permanently favor "face"). This supports models positing that pareidolia reflects rapid activation of face-detection circuits by face-like visual features, rather than a fixed propensity to see faces[1][2].

2. **Task instructions modulate decision caution (*a*), not evidence strength (*v*)**, showing that observers can strategically adjust when they commit to a "face" judgment based on instructions, even though the visual evidence itself (pareidolia) remains ambiguous. This refines the theory of speed-accuracy trade-offs in perception.

3. **Base-rate expectations bias the decision starting point (*z*), enabling Bayesian-like updating of face priors.** This suggests that face-detection circuits integrate prior probability information in a manner consistent with rational statistical inference[citation needed: Bayesian decision theory account of starting-point bias in perceptual decisions. Expected evidence: prior DDM paper linking *z* to log-prior odds of target category].

4. **Pareidolia sits between real faces and non-faces in evidence strength**, validating the stimulus hierarchy and the Goldilocks-zone framework[1]. Individual variation in *v*_Pareidolia may index susceptibility to face illusions across contexts (design, art, nature).

### Theoretical Implications

These findings align with hierarchical models of face processing, wherein early retinotopic regions encode low-level features, mid-level regions (e.g., OFA) aggregate those features into face-tuned representations, and high-level regions (e.g., FFA) achieve categorical face identity[2]. The rapid (~160 ms) FFA response to pareidolia[2] likely reflects feed-forward propagation of sufficient visual evidence to activate face-selective neurons, without yet committing the decision system to a conscious "face" judgment. By decomposing RT/accuracy into *v* and *a*, the present study bridges this neural timeline and the behavioral decision boundary.

### Clinical and Applied Implications

Understanding pareidolia as a manipulable perceptual phenomenon with separable components (*v*, *a*, *z*) opens avenues for clinical investigation. For example:

- **Schizophrenia**: Elevated pareidolia in schizophrenia[4] could reflect heightened *v*_Pareidolia (hyperactive face detection), inflated *z* (face-biased prior), or reduced *a* (impulsive commitment to face judgments). DDM decomposition could specify the mechanism, informing cognitive remediation[citation needed: proof-of-concept study using DDM to characterize perceptual decision deficits in schizophrenia and predict response to training. Expected evidence: intervention study showing that training to adjust *a* or *z* reduces paranoia or psychotic symptoms in patient populations].

- **Anxiety**: Individuals with anxiety may show heightened *v* for threat-relevant pareidolic patterns (e.g., faces in crowds during social anxiety)[citation needed: theoretical link between anxiety and face-detection bias via decision parameters. Expected evidence: prior study showing that social anxiety correlates with either *v* or *z* bias toward faces in ambiguous-stimulus tasks].

- **Design and user experience**: The Goldilocks zone for pareidolia[1] has implications for product design; understanding the DDM parameters underlying pareidolia could help designers either maximize or minimize unintended face illusions in product surfaces.

### Limitations of the Present Study

1. **Sample characteristics**: Healthy young adults (18–35) may not generalize to older adults, clinical populations, or diverse cultures. Future work should replicate with clinical samples (schizophrenia, anxiety) and cross-cultural cohorts.

2. **Stimulus size and presentation**: 400 × 400 pixel images at a fixed viewing distance may not capture pareidolia in natural scenes (landscapes, architectural elements) or at varying distances. Ecological validity is limited.

3. **Single session, no longitudinal data**: This is a cross-sectional snapshot. Individual differences in pareidolia susceptibility may fluctuate with fatigue, context, or mood; longitudinal designs are needed to establish stability.

4. **Correlational personality analysis**: Extraversion is explored correlatively; no manipulations of motivation or social context are included. Causal inferences are not warranted[citation needed: discussion of correlational vs. experimental inference in personality-cognition links. Expected evidence: methodology paper distinguishing correlational and experimental approaches to understanding personality-decision relationships].

5. **Model assumptions**: Standard DDM assumes independent evidence accumulation and no strategic switching between decision criteria mid-trial. Pareidolia might involve post-hoc reinterpretation (late cognitive override), violating these assumptions. Relaxing these assumptions (e.g., allowing criterion shifts) would require more complex models and larger datasets.

---

## Limitations & Risk Mitigation

### Design and Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Insufficient trial accumulation (low accuracy, high RTs) due to task difficulty | Moderate | High | Pilot N=5–8 participants; adjust stimulus clarity or block difficulty based on feedback. Include practice trials and difficulty adjustments in Speed vs. Accuracy blocks. |
| MCMC non-convergence in HDDM fitting | Low–Moderate | High | (1) Implement multiple parallel chains and extended sampling. (2) Use weakly informative priors. (3) Fallback to maximum-likelihood DDM if Bayesian fails. (4) Reduce model complexity (e.g., remove Evidence × Instruction interaction). |
| Confounding of base-rate and stimulus composition | Low | Moderate | Ensure that stimulus allocation (Real : Pareidolia : Non-face ratio) is identical across High-BR and Low-BR blocks; only the face-category proportion (Real + Pareidolia) varies. Include pre-experimental checks (pilot stimuli). |
| Individual differences in task compliance (e.g., some participants ignore speed instructions) | Moderate | Moderate | Monitor RT distributions and accuracy within-block; provide verbal feedback after Speed block (e.g., "Your RTs in the Speed block were X ms; maintain accuracy!"). |
| Generalization of base-rate effect: participants may not fully adopt the statistical structure | Moderate | Low–Moderate | Include an embedded "catch" trial block where base-rate flips abruptly mid-block; measure the lag in *z* adjustment. Report individual sensitivity to base-rate in supplementary analyses. |

### Analytical Risks

| Risk | Mitigation |
|------|-----------|
| Multiple comparisons and Type I error in exploratory tests | Use Bayesian model comparison with LOO-IC (avoiding traditional NHST thresholds). Treat personality correlations as exploratory; pre-specify focal hypotheses (H1–H5) and test these with higher confidence than secondary effects. |
| Posterior predictive checks fail to constrain model | If simulated data show systematic discrepancies from observed, consider re-specifying model (e.g., adding *τ_v* across-trial variability, or allowing *t₀* to vary by Evidence). Document all model modifications in preregistration amendments. |
| Individual-level parameters poorly recovered | At N=35, some participants may yield unreliable parameter estimates. Use hierarchical shrinkage to stabilize estimates; compute individual prediction intervals (Bayesian credible intervals) and report with confidence qualifications (e.g., "High uncertainty for participant 12.") |

---

## Open Science & Reproducibility Plan

### Preregistration

This protocol will be preregistered at **OSF (Open Science Framework)** prior to data collection, including:
- Hypotheses (H1–H5) and their DDM parameter mappings.
- Data exclusion criteria (RT <200 ms, >5000 ms; accuracy <55%).
- Primary and secondary analyses, with model formulas.
- Planned model comparisons and decision rules (e.g., elpd_diff ≥ 4 for model preference).

Amendments will be logged with timestamps if analytical changes are needed (e.g., due to convergence failures).

### Data and Code Availability

Upon publication, we will release:
- **Raw behavioral data**: Reaction times and accuracy per trial, with participant and condition labels (with data masked for privacy).
- **HDDM model code**: Fully annotated Python script (using PyMC3 and HDDM packages) reproducing all posterior inferences.
- **Stimulus images** (pareidolia, real faces, non-faces): Made available via OSF (ensuring copyright compliance or using public-domain images).
- **Analysis scripts**: R/Python scripts for posterior predictive checks, model comparison, and plotting.

All code and data will be deposited in a public repository (OSF or GitHub) with a DOI.

### Transparency and Corrections

We commit to:
- Reporting all tested hypotheses, including null findings, in the final manuscript.
- Distinguishing pre-registered predictions from exploratory post-hoc analyses.
- Disclosing any analysis deviations and their rationale.
- If results substantially contradict predictions, providing detailed discussion of plausible explanations (e.g., model misspecification, stimulus factors).

---

## References

Avery, J., Smith, P., & Spaniol, J. (2023). Accounting for endogenous effects in decision-making with a non-linear drift diffusion model. *PLoS Computational Biology*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10113207/

Liu, Y., et al. (2021). Pareidolia in schizophrenia and bipolar disorder. *Frontiers in Psychiatry, 12*, 746734. https://doi.org/10.3389/fpsyt.2021.746734

Ward, J., & Smith, A. (2024). Seeing faces in things: A model and dataset for pareidolia. *arXiv preprint arXiv:2409.16143*. https://arxiv.org/html/2409.16143v1

Ward, J., Conkar, D., & Quinn, F. (2020). Rapid and dynamic processing of face pareidolia in the human brain. *Nature Communications, 11*(1), 4528. https://doi.org/10.1038/s41467-020-18325-8

---

## Summary Note

This protocol specifies a feasible 20-minute in-lab study (N=35, ~180 trials) applying hierarchical Bayesian DDM to decompose pareidolia decisions under systematic manipulation of evidence strength, decision instructions, and base-rate expectations. The design directly tests whether pareidolia reflects ambiguous perceptual evidence (*v*), decision bias (*z*), decision caution (*a*), or combinations thereof, grounding three-level stimulus hierarchies in validated pareidolia datasets and neural evidence[1][2]. Primary limitations include modest sample size, single-session design, and restricted stimulus ecology; mitigation strategies include preregistration, open-science practices, and transparent reporting of limitations. This study exemplifies how computational cognitive modeling can isolate mechanistic origins of perceptual illusions and individual differences.