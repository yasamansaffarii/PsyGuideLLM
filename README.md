# PsyGuide-LLM

### Theory-Guided Contrastive Fine-Tuning of LLM-Derived Psychological Embeddings for Interpretable Burnout and Attrition-Risk Prediction, with Transfer to Person–Job Fit Scoring

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)]()
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

> **PsyGuide-LLM** is a theory-guided representation learning framework in which psychological constructs extracted from employee text are incorporated into contrastive representation learning. The resulting embeddings are designed to capture theoretically meaningful psychological structure while supporting interpretable burnout and attrition-risk prediction.

**Paper:** *Manuscript in preparation / to be submitted*
**Code:** This repository
**Status:** Research prototype; final experimental results are being prepared for publication.

---

## Overview

Employee reviews and other forms of organizational text contain information that may reflect work-related stress, disengagement, personality-related tendencies, and dissatisfaction. Conventional text classifiers, however, are primarily optimized for predictive performance and are not explicitly constrained by psychological theory.

PsyGuide-LLM addresses this gap by combining:

* **Large Language Models (LLMs)** for psychological construct extraction,
* **Big Five personality theory** for personality-related representations,
* **Maslach Burnout Inventory (MBI)** dimensions for work-related burnout signals,
* **contrastive representation learning** for embedding adaptation,
* **LoRA-based parameter-efficient fine-tuning**,
* and a **Theory-Guided Contrastive Loss (TGCL)** that incorporates continuous psychological similarity into the representation-learning objective.

Rather than treating psychological scores as ground-truth psychological assessments, they are used as **auxiliary theory-space supervision** for organizing the representation space.

---

## Research Question

The central question investigated by PsyGuide-LLM is:

> **Can theoretically grounded psychological information extracted from organizational text be used to improve the learned representation space for burnout and attrition-risk prediction beyond conventional text representations and label-only contrastive learning?**

The framework is evaluated through controlled comparisons designed to separate:

1. the contribution of psychological information,
2. the contribution of contrastive pretraining,
3. the contribution of the *theoretical structure* imposed by psychological constructs,
4. and the contribution of the learned representation itself.

---

## Method

The complete pipeline consists of five main stages.

```text
Employee Text
     │
     ▼
┌──────────────────────────────┐
│ LLM-based Construct          │
│ Extraction                   │
│                              │
│ Big Five + MBI               │
└──────────────┬───────────────┘
               │
               ▼
      Psychological Vector z
               │
               │
Employee Text ─┴───────────────┐
                               ▼
                 ┌────────────────────────┐
                 │ Theory-Guided           │
                 │ Contrastive Learning    │
                 │                        │
                 │ MiniLM + LoRA           │
                 │ + TGCL                  │
                 └────────────┬───────────┘
                              │
                              ▼
                   Theory-Guided Embedding
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
       Burnout / Attrition             Person–Job Fit
          Risk Prediction                 Ranking
                │
                ▼
       Interpretability & Fairness
```

---

## 1. Psychological Construct Extraction

Eight psychological dimensions are extracted from text using an instruction-tuned LLM:

### Big Five Personality Dimensions

* **Extraversion**
* **Neuroticism**
* **Agreeableness**
* **Conscientiousness**
* **Openness**

### MBI-Related Burnout Dimensions

* **Exhaustion**
* **Cynicism**
* **Reduced Professional Efficacy**

The construct definitions are operationalized using established psychological frameworks rather than generic LLM prompts.

In particular, the implementation distinguishes **Cynicism in the MBI-General Survey** from **Depersonalization in the MBI-Human Services Survey**, avoiding a direct conflation of the two constructs.

---

## 2. Chain-of-Constructs Prompting

Psychological constructs are extracted using a structured **Chain-of-Constructs prompting protocol**.

Instead of asking the LLM to produce a single undifferentiated psychological assessment, each construct is evaluated separately using:

* an operational definition,
* linguistic indicators,
* construct-specific instructions,
* and repeated sampling when self-consistency is enabled.

The final construct score is obtained from the generated estimates.

The extraction process is implemented with **batched generation** to reduce computational cost, and extracted representations can be cached to disk or Google Drive to support reproducible Colab execution.

---

## 3. Theory-Guided Contrastive Loss

The central methodological component is the **Theory-Guided Contrastive Loss (TGCL)**.

For a pair of texts \(i,j\), similarity is defined using both:

1. agreement in the task label, and
2. similarity in the extracted psychological theory space.

The target similarity is therefore defined as a weighted combination:

$$
s_{ij}^{target}
=
\lambda s_{ij}^{theory}
+
(1-\lambda)s_{ij}^{label}
$$

where:

* \(s_{ij}^{theory}\) represents similarity between the psychological construct vectors,
* \(s_{ij}^{label}\) represents label-based similarity,
* and \(\lambda\) controls the contribution of theory-guided supervision.

This formulation provides a direct connection between psychological structure and representation learning.

### SupCon as a special case

When:

$$
\lambda = 0
$$

the objective becomes the corresponding **Supervised Contrastive Learning** formulation.

Therefore, SupCon is implemented as a special case of the same training procedure rather than as a separately implemented baseline.

---

## 4. Parameter-Efficient Fine-Tuning

The text encoder is initialized from:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and is adapted using **LoRA (Low-Rank Adaptation)**.

This design provides a relatively lightweight representation-learning pipeline while allowing the encoder to be optimized for the theory-guided objective.

The contrastively adapted encoder is subsequently combined with the explicit psychological representation for the downstream prediction task.

---

## 5. Fusion-Based Risk Prediction

For the final prediction stage, the learned text representation is combined with the eight explicit psychological construct scores.

The resulting representation is passed to a **LightGBM classifier**.

```text
TGCL Embedding
      +
8 Psychological Scores
      │
      ▼
   LightGBM
      │
      ▼
Burnout / Attrition-Risk Score
```

This fusion design allows both the learned semantic representation and the explicit psychological dimensions to contribute to prediction.

It also enables construct-level interpretation of the final prediction.

---

# Experimental Design

A controlled ablation framework is used to determine whether the proposed method provides value beyond conventional text representations and simple psychological features.

The main comparison contains **nine models**.

| Model                 | Representation / Training Strategy           |
| --------------------- | -------------------------------------------- |
| TF-IDF + LR           | Classical lexical representation             |
| Frozen MiniLM + LR    | Frozen sentence embeddings                   |
| Fine-tuned DistilBERT | Fully fine-tuned transformer                 |
| Zero-shot LLM         | Direct LLM classification                    |
| Psych-only            | Eight psychological scores                   |
| Text + Psych          | Frozen MiniLM + psychological scores         |
| SupCon-Fusion         | Label-only contrastive learning              |
| Shuffled-TGCL-Fusion  | TGCL with permuted theory information        |
| **TGCL-Fusion**       | **Proposed theory-guided contrastive model** |

The main models are evaluated on a **common evaluation subset**, ensuring that methods requiring expensive LLM-based psychological extraction are compared against the same examples used by the computationally cheaper baselines.

---

# Hypotheses

Three hypotheses are defined before the final evaluation.

### H1 — Value of Contrastive Pretraining

$$
AUROC(TGCL\text{-}Fusion)
>
AUROC(Text+Psych)
$$

This hypothesis tests whether theory-guided contrastive representation learning provides additional value beyond simply concatenating psychological features with a text representation.

### H2 — Theory vs. Label-Only Contrastive Learning

$$
AUROC(TGCL\text{-}Fusion)
>
AUROC(SupCon\text{-}Fusion)
$$

This hypothesis tests whether psychological theory provides information beyond the task labels used by conventional supervised contrastive learning.

### H3 — Theory Content vs. Structural Regularization

$$
AUROC(TGCL\text{-}Fusion)
>
AUROC(Shuffled\text{-}TGCL\text{-}Fusion)
$$

A fixed permutation of psychological representations is used in the control condition.

This comparison tests whether performance depends on the **actual correspondence between texts and psychological constructs**, rather than merely on the presence of an additional continuous signal.

---

# Hyperparameter Selection

The theory contribution is controlled through a validation-based \(\lambda\)-sweep.

For the full experimental run:

```text
λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

The selected value \(\lambda^*\) is determined using validation AUROC.

The test set is not used for selecting \(\lambda\), reducing the risk of test-set leakage.

---

# Datasets

## Essays

The **Essays** dataset is used to evaluate the LLM-based Big Five construct extraction against existing personality annotations.

* 2,467 essays
* Five Big Five personality labels
* Used as a validation resource for the psychological extraction stage

The dataset is loaded directly from the publicly available repository used by the notebook.

## Glassdoor Job Reviews

The main prediction experiments are based on the **Glassdoor Job Reviews** dataset.

The dataset contains large-scale employee reviews including free-text information such as employee-reported pros and cons and recommendation-related information.

A balanced subset is constructed for the prediction experiments.

The target variable is treated as a **proxy for employee dissatisfaction / attrition risk**, rather than as a direct measurement of observed employee turnover.

This distinction is important when interpreting the results.

## Resume / Job-Description Data

A resume/job-description dataset is used for a qualitative **Person–Job Fit** case study.

This component is intended as a demonstration of transfer rather than as a fully supervised hiring benchmark.

---

# Person–Job Fit Transfer

The learned TGCL representation is also evaluated in a secondary application:

> **Can the theory-guided embedding provide more psychologically meaningful Person–Job Fit rankings than a generic frozen sentence embedding?**

Candidate descriptions and job descriptions are embedded using:

* the proposed TGCL encoder,
* and the frozen MiniLM baseline.

Similarity scores are then used to rank candidate–job pairs.

This component is intentionally presented as a **qualitative case study**, since publicly available datasets with reliable ground-truth Person–Job Fit labels are limited.

No claim of validated hiring effectiveness is made from this component alone.

---

# Interpretability

Interpretability is examined using **SHAP** on the downstream LightGBM fusion model.

Because the final classifier receives both:

* learned text embeddings,
* and explicit psychological dimensions,

the contribution of individual psychological constructs can be examined.

For example, the analysis can be used to investigate whether dimensions such as:

* Exhaustion,
* Cynicism,
* Reduced Professional Efficacy,
* Neuroticism,
* or Conscientiousness

contribute strongly to predicted risk.

This provides a more explicit interpretation layer than a prediction based solely on an opaque text embedding.

---

# Fairness Analysis

A subgroup-level fairness analysis is included where suitable subgroup information is available in the dataset.

Performance can be compared across relevant employee groups to identify potential disparities in predictive behavior.

Fairness analysis is treated as a diagnostic component rather than as evidence that the model is automatically fair or suitable for deployment in employment decisions.

---

# Statistical Evaluation

The main evaluation reports:

* **AUROC**
* **F1**
* **Balanced Accuracy**

For pairwise comparisons, the notebook includes:

* paired bootstrap confidence intervals,
* AUROC difference testing,
* and McNemar's test for paired classification outcomes.

The three primary hypotheses are evaluated on the common evaluation subset.

The intended decision criterion is based on both statistical significance and the direction of the observed effect.

---

# Reproducibility

The notebook provides two execution modes.

### Quick Test

```python
QUICK_TEST = True
```

The quick configuration uses substantially smaller subsets and fewer training iterations.

It is intended for:

* debugging,
* dependency checking,
* pipeline validation,
* and detecting implementation errors.

It **must not be interpreted as the final experimental configuration**.

### Full Run

```python
QUICK_TEST = False
```

The full configuration increases:

* Glassdoor sample size,
* Essays validation size,
* contrastive training pool,
* evaluation subset,
* DistilBERT training size,
* self-consistency sampling,
* contrastive epochs,
* and the \(\lambda\)-search grid.

The final numerical results reported in the accompanying paper should be generated from this configuration.

---

# Hardware

A GPU runtime is recommended for the full experiment.

The notebook has been designed for Google Colab and supports conditional CPU/GPU execution.

For the full run, a GPU runtime such as:

```text
Google Colab T4
```

is recommended.

The most computationally expensive component is LLM-based psychological construct extraction.

Caching is therefore provided to reduce repeated extraction across sessions.

---

# Installation

The main dependencies are installed with:

```bash
pip install -q \
    transformers \
    accelerate \
    peft \
    bitsandbytes \
    sentence-transformers \
    datasets \
    kagglehub \
    lightgbm \
    shap \
    scikit-learn \
    scipy \
    statsmodels \
    tqdm
```

The notebook also includes a compatibility step for `peft` and `torchao` in the default Colab environment.

---

# Running the Project

The complete pipeline is provided in:

```text
PsyGuideLLM_colabfinal.ipynb
```

### 1. Open the notebook in Google Colab

### 2. Select a runtime

For initial validation:

```text
Runtime → Change runtime type → CPU
```

For the full experiment:

```text
Runtime → Change runtime type → T4 GPU
```

### 3. Run the quick test

Set:

```python
QUICK_TEST = True
```

and execute the notebook from the beginning.

### 4. Run the full experiment

After the complete quick test has been executed successfully:

```python
QUICK_TEST = False
```

Then rerun the notebook using a GPU runtime.

### 5. Kaggle authentication

The Glassdoor dataset requires Kaggle access.

A Kaggle API token (`kaggle.json`) is required by the notebook to download the dataset.

### 6. Optional Google Drive caching

Google Drive can be connected to preserve the expensive LLM extraction cache across Colab sessions.

---

# Current Experimental Status

The repository should currently be regarded as a **research prototype accompanying a manuscript in preparation**.

The notebook contains the complete experimental pipeline, including:

* psychological construct extraction,
* construct validation,
* multiple text baselines,
* psychological-feature baselines,
* contrastive representation learning,
* TGCL,
* SupCon ablation,
* shuffled-theory control,
* validation-based \(\lambda\)-selection,
* statistical testing,
* SHAP analysis,
* fairness analysis,
* and Person–Job Fit transfer.

Importantly, the final paper results should be generated using the full experimental configuration.

Preliminary `QUICK_TEST` outputs are retained in the notebook for pipeline verification and should **not** be interpreted as the final reported results.

---

# Methodological Positioning

The framework is intended to investigate a specific methodological question rather than simply to maximize predictive performance.

The key distinction is made between:

```text
Text similarity
      vs.
Task-label similarity
      vs.
Theory-grounded psychological similarity
```

Conventional supervised contrastive learning primarily uses the second source of information.

PsyGuide-LLM introduces the third source by incorporating continuous similarity in a theoretically defined psychological space.

The experimental design is therefore centered on the following progression:

```text
Text-only
   ↓
Text + Psychological Features
   ↓
Label-based Contrastive Learning
   ↓
Theory-Guided Contrastive Learning
   ↓
Shuffled-Theory Control
```

This progression is intended to isolate the contribution of theory-guided representation learning.

---

# Limitations

Several limitations should be considered when interpreting the framework.

### 1. Psychological scores are model-derived

The extracted Big Five and MBI-related scores are generated by an LLM and should not be interpreted as clinical or psychometric assessments of individual employees.

They are used as auxiliary representations for machine learning.

### 2. Attrition is indirectly measured

The Glassdoor target is used as a proxy for dissatisfaction / attrition risk.

It should not be interpreted as direct observation of actual employee turnover.

### 3. Person–Job Fit is currently qualitative

The Person–Job Fit component is a demonstration rather than a validated hiring benchmark.

A future evaluation with expert HR judgments or an appropriately labeled dataset would provide stronger evidence.

### 4. Employment-domain deployment requires additional validation

The framework is research-oriented.

Before any real-world use in hiring, promotion, retention, or employee assessment, substantial validation would be required, including legal, ethical, psychometric, privacy, and fairness assessments.

### 5. LLM extraction introduces additional uncertainty

Prompting, model choice, sampling, and self-consistency can affect psychological construct estimates.

The extraction-validation stage is therefore an important part of the proposed pipeline.

---

# Reproducibility Checklist

Before reporting results in the paper, the following should be completed:

* [ ] Run the complete `QUICK_TEST` pipeline without errors.
* [ ] Run the complete `FULL_RUN` configuration.
* [ ] Record the exact dataset version.
* [ ] Record the exact LLM and encoder versions.
* [ ] Record the selected \(\lambda^*\).
* [ ] Report final AUROC, F1, and Balanced Accuracy.
* [ ] Report bootstrap confidence intervals.
* [ ] Report H1, H2, and H3 results.
* [ ] Report the shuffled-theory control.
* [ ] Report psychological extraction validation results.
* [ ] Generate SHAP analyses.
* [ ] Run the intended fairness analysis.
* [ ] Clearly distinguish proxy outcomes from directly observed outcomes.

---

# Theoretical Foundations

The psychological construct definitions are grounded in established literature, including:

* Maslach, C., Schaufeli, W. B., & Leiter, M. P. (2001). *Job burnout*. Annual Review of Psychology, 52, 397–422.
* Maslach, C., & Jackson, S. E. (1981). *The measurement of experienced burnout*. Journal of Organizational Behavior, 2(2), 99–113.
* Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). *Maslach Burnout Inventory Manual*.
* Costa, P. T., Jr., & McCrae, R. R. (1992). *Four ways five factors are basic*. Personality and Individual Differences, 13(6), 653–665.
* McCrae, R. R., & Costa, P. T., Jr. (1987). *Validation of the five-factor model of personality across instruments and observers*. Journal of Personality and Social Psychology, 52(1), 81–90.
* McCrae, R. R., & John, O. P. (1992). *An introduction to the five-factor model and its applications*. Journal of Personality, 60(2), 175–215.

The contrastive-learning component is implemented as a theory-guided extension of supervised contrastive representation learning.

---

# Citation

The associated manuscript is currently being prepared for submission.

Once the paper has been accepted, the citation below will be updated with the final bibliographic information.

```bibtex
@article{psyguide_llm,
  title   = {PsyGuide-LLM: Theory-Guided Contrastive Fine-Tuning of LLM-Derived Psychological Embeddings for Interpretable Burnout and Attrition-Risk Prediction},
  author  = {Author Names},
  journal = {Venue},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

# Disclaimer

PsyGuide-LLM is provided for **research purposes only**.

The framework is not intended to provide clinical psychological assessment, diagnosis, or employment decisions. Model-derived psychological representations should not be treated as validated assessments of individual employees.

---

## Acknowledgements

The project builds upon established work in organizational psychology, personality modeling, burnout research, transformer-based language representation learning, parameter-efficient fine-tuning, and contrastive representation learning.

---

## Contact

For questions, research collaboration, or discussion of the methodology, please open an issue in this repository or contact the authors through the information provided in the associated manuscript.
