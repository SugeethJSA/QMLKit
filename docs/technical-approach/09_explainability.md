# 09 — Explainability

`src/qmlkit/explainability/` — two components that turn a model output into a
clinically-shaped attribution.

| Component | File | Purpose |
|---|---|---|
| `BiomarkerAttributionEngine` | `biomarker_mapper.py` | latent qubit attributions → VOC compounds → biochemical pathways → prose |
| `QuantumKernelSHAP` | `quantum_shap.py` | exact Shapley values over the reduced feature space, via `shap.KernelExplainer` |

This layer was rewritten as part of the code-evaluation pass because the original
hardcoded the synthetic 16-sensor/24-compound geometry and would have produced
**misleading attributions on the real 27-column VOC dataset** (finding #2, High
— `docs/reports/code_evaluation_report.md:21`).

---

## 1. `BiomarkerAttributionEngine`

### 1.1 Construction

```python
BiomarkerAttributionEngine(
    feature_selector: QuantumFeatureSelector,
    voc_config: Optional[VOCBiomarkerConfig] = None,   # defaults to 24 compounds
    feature_names: Optional[List[str]] = None,          # names of the selector's INPUT features
)
```

`feature_names` is the key to correct behaviour on real data: when the selector's
input columns *are* the configured compounds, attributions map **one-to-one** and
no pooling heuristic is needed.

### 1.2 `map_latent_to_chemical` — latent → raw features

```python
def map_latent_to_chemical(self, latent_shap_values):
    latent_vec = np.atleast_2d(latent_shap_values)
    if self.feature_selector.method == "pca" and self.feature_selector.pca_model is not None:
        sensor_shap = np.dot(latent_vec, self.feature_selector.pca_model.components_)
        return self._pool_to_compounds(sensor_shap)
    else:
        # Fallback uniform projection
        return np.ones((latent_vec.shape[0], n_comp)) / n_comp      # ← DEGENERATE
```

**PCA case:** `components_` has shape `(n_components, n_input_features)`, so
`latent_shap @ components_` is exactly the linear back-projection. This is
mathematically exact for PCA and is the reason `inverse_transform` is only
supported for `pca` and `autoencoder` in `QuantumFeatureSelector`.

**Non-PCA case: returns a uniform matrix.** Every compound gets
$1/n_\text{comp}$ regardless of the input. This is a **constant**, not an
explanation — a `mutual_info` or `autoencoder` selector produces a
`BiomarkerWaterfall` where all 24 bars are identical. It does not raise, so the
degradation is silent. For `mutual_info` an exact back-projection would be
trivial (scatter the 6 selected latents into a 27-vector) and is the obvious
improvement.

### 1.3 `_pool_to_compounds` — three-tier strategy

`biomarker_mapper.py:41-71`.

**Tier 1 — direct mapping (`:47-50`)**

```python
if self.feature_names is not None and n_feat == n_comp and self.feature_names == list(self.compounds):
    return feature_shap
```

Correct and exact, but requires the input features to be *named* exactly as the
24 configured compounds. The bundled real dataset's columns are molecular
formulas (`CH2O`, `C2H4O`, …), not compound names, so **tier 1 does not fire on the
real VOC data** — see §1.5.

**Tier 2 — legacy synthetic schema (`:54-63`)**

```python
if n_feat == 64 and n_comp == 24:
    for c_idx in range(24):
        block = c_idx // 6                                  # 4 chemical-class blocks
        sensor_indices = [block + 4*k for k in range(4)]     # 4 sensors per block
        feat_indices = [s*4 + f for s in sensor_indices for f in range(4)]
        chem_shap[:, c_idx] = np.mean(feature_shap[:, feat_indices], axis=1)
```

Mean-pools 4 sensors × 4 kinetic features = 16 input features per compound. This
encodes the generator's physical assumption: sensor $s$ is preferentially
sensitive to chemical class $s \bmod 4$, with 4 sensors per class and 4 kinetic
descriptors each.

**Tier 3 — generic contiguous pooling (`:65-71`)**

```python
edges = np.linspace(0, n_feat, n_comp + 1, dtype=int)
for c_idx in range(n_comp):
    lo, hi = edges[c_idx], max(edges[c_idx] + 1, edges[c_idx + 1])
    chem_shap[:, c_idx] = np.mean(feature_shap[:, lo:hi], axis=1)
```

Equal-count contiguous bins with a `max(lo+1, hi)` guard so no compound gets an
empty slice. For 27 features → 24 compounds this means the first 21 compounds get
1 feature each and the last 6 get 1 feature each from overlapping edges — a
**positional heuristic with no chemical meaning**. This is the tier that fires on
the real VOC dataset.

### 1.4 `generate_explanation` — latent → clinical report

`biomarker_mapper.py:90-150`.

**Pathway aggregation** (`:102-118`):

```python
pathway_names = ["Lipid_Peroxidation_Aldehydes",
                 "Mitochondrial_Ketone_Metabolism",
                 "Cytochrome_P450_Aromatics",
                 "Alkanes_Sulfur_Dysregulation"]
edges = np.linspace(0, 24, 5, dtype=int)          # 0,6,12,18,24
block_sums[k] = np.sum(np.maximum(0, chem_shap[edges[k]:edges[k+1]]))   # POSITIVE ONLY
total_pos = max(1e-6, sum(block_sums))
pathways[name] = round(block / total_pos * 100, 1)
```

Two properties to be aware of:

1. **Only positive contributions count.** `np.maximum(0, ·)` means protective /
   depleted compounds contribute 0 to their pathway. If all attributions are
   negative, `total_pos` collapses to the `1e-6` floor and all four pathways
   report 0.0%.
2. **The four block boundaries are the compound list's own class structure.**
   This is chemically correct for the 24-compound config (4 blocks × 6) and
   arbitrary for any other compound ordering. The `VOCBiomarkerConfig` docstring
   and the comment at `:100-101` both make the coupling explicit.

**Top biomarkers** (`:122-132`):

```python
for idx in np.argsort(np.abs(chem_shap))[::-1][:5]:
    impact = float(chem_shap[idx])
    direction = "Elevated (Risk Driver)" if impact > 0 else "Depleted / Protective"
    {"compound": compounds[idx], "importance_score": round(abs(impact), 4),
     "clinical_impact": direction}
```

Always exactly 5, ranked by |attribution|.

**Output strings** (`:134-141`):

```python
pred_class = "Cancer Positive (High Risk)" if p >= 0.5 else "Healthy / Low Risk"
summary = (f"Patient {sample_id} evaluated with {p*100:.1f}% malignancy risk. "
           f"The primary biochemical driver was "
           f"{pathways['Lipid_Peroxidation_Aldehydes']}% lipid peroxidation "
           f"with significant elevation in [{top3 joined}].")
```

Note the summary **hardcodes the lipid-peroxidation pathway as "the primary
biochemical driver"** even when that pathway contributes 0% and a different
pathway dominates. The `frontends`' `BiomarkerWaterfall` renders the actual four
percentages, so the discrepancy is visible on screen — but the prose is wrong in
that case. Fixable in one line (`max(pathways, key=pathways.get)`).

### 1.5 Honest assessment on the real dataset

| Stage | Real VOC (27 formula columns) | Synthetic (64 sensor-kinetic) |
|---|---|---|
| latent → raw | **exact** (PCA linear back-projection) | exact |
| raw → compound | **tier 3, positional heuristic** | tier 2, physically motivated |
| compound → pathway | positional (4 × 6 blocks) | positional (4 × 6 blocks) — matches chemistry |
| non-PCA selector | **uniform constant** | uniform constant |

So on the real data the *magnitude* of the attributions is meaningful (it comes
out of a real SHAP/PCA computation) but the *compound labels* attached to them
are positional. `server.py` does not pass `feature_names`, so tier 1 never fires.
The presentation should say: *"attribution magnitudes are computed from the
model; compound-level naming currently uses a positional mapping on the real
dataset and will be replaced with a curated formula→compound dictionary."*

### 1.6 Tests

`tests/test_explainability.py` (1 test): 30×64 data, PCA selector, 6-dim latent
SHAP → `(1, 24)` compound matrix; `generate_explanation` echoes the `sample_id`,
yields `"Cancer Positive"` for $p \ge 0.5$, exactly 5 top biomarkers, and includes
`Lipid_Peroxidation_Aldehydes` in the pathway dict with a >20-char summary.

Not tested: tiers 1 and 3 of `_pool_to_compounds`, the non-PCA uniform fallback,
the `total_pos` floor branch, and the case where a different pathway dominates.

---

## 2. `QuantumKernelSHAP`

`quantum_shap.py:10-59`.

### 2.1 Construction

```python
def __init__(self, predict_fn, background_data, n_samples_background=25):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("QuantumKernelSHAP requires the optional 'shap' package. "
                          "Install it via `pip install shap` or `pip install qmlkit[dev]`.") from exc
    if len(background_data) > n_samples_background:
        indices = np.random.default_rng(42).choice(len(background_data),
                                                    size=n_samples_background, replace=False)
        bg = background_data[indices]
    else:
        bg = background_data
    self.explainer = shap.KernelExplainer(self._predict_wrapper, bg)
```

- **Lazy import with an actionable error** — was a top-level `import shap` that
  broke `import qmlkit.explainability` on minimal installs (finding #7, Medium).
  Fixed.
- Background subsampling is **seeded at 42** and without replacement.

### 2.2 Why `KernelExplainer`

`shap.KernelExplainer` computes Shapley values by **weighted linear regression**
over sampled coalitions. This is the only SHAP explainer that works with an
arbitrary black-box `predict_fn` — the model-agnostic path. `shap.TreeExplainer`
would be far faster but requires a tree ensemble; the QSVM/VQC/QCNN are not
trees. The cost is $O(2^K \cdot n_\text{evals})$ model evaluations for $K$
features, which for the quantum heads means $O(2^6 \cdot 48) = 3{,}072$ QNode
sweeps per sample at `n_evals=48` — expensive, and only enabled by
`?deep_explain=true`.

It is a reasonable choice for "explain a QSVM", and the cost is the reason the
fast path exists (§3).

### 2.3 `_predict_wrapper` and `explain`

```python
def _predict_wrapper(self, x):
    probs = self.predict_fn(x)
    if probs.ndim == 1:      return np.vstack([1.0 - probs, probs]).T
    elif probs.shape[1] == 2: return probs[:, 1]      # positive-class probability
    return probs

def explain(self, x_instance, n_evals=150):
    x_mat = np.atleast_2d(x_instance)
    shap_vals = self.explainer.shap_values(x_mat, nsamples=n_evals, silent=True)
    if isinstance(shap_vals, list):
        return np.asarray(shap_vals[1])               # binary → [class0, class1]
    return np.asarray(shap_vals)
```

Attribution is computed on the **positive-class probability**, so SHAP values sum
to $f(x) - \mathbb{E}[f]$ — interpretable as "how much does this push toward
cancer".

### 2.4 Status

**Not covered by any test** and not exported from
`explainability/__init__.py`. It is instantiated in `server.py:85-93` inside a
`try/except ImportError`, and `shap` is in the **dev extra** only — so on a
minimal install `shap_explainer` is `None` and the deep path silently degrades.

---

## 3. How the API wires explainability

`api/server.py:152-191`, `POST /api/v1/predict?deep_explain=`:

```python
q_feat = selector.transform(X_scaled)
preds   = qsvm.predict(q_feat)
proba   = qsvm.predict_proba(q_feat)
pos     = float(proba[0, 1])

if deep_explain and shap_explainer is not None:
    try:
        latent_shap = shap_explainer.explain(q_feat, n_evals=48)[0]
    except Exception:
        latent_shap = None
else:
    latent_shap = q_feat[0] - np.mean(q_feat)          # fast mean-delta
```

**Two attribution quality levels:**

| Mode | Method | Cost | Meaning |
|---|---|---|---|
| default (`deep_explain=false`) | `q_feat[0] - np.mean(q_feat)` | free | a sample's deviation from the training mean in angle space — **a descriptive statistic, not a model attribution** |
| `deep_explain=true` | kernel SHAP, 48 evals | ~3,000 QNode sweeps | true Shapley values on the model output |

The fast path is honest in what it is, but the API does not label which one was
used — a client cannot tell from the response whether it received a mean-delta or
a Shapley vector. Adding a `"attribution_method": "shap"|"mean_delta"` field to
the response would resolve this; it is a one-line change.

**Response shape** (`server.py:152-191`):

```json
{
  "sample_id": "SMPL_ONC_8892",
  "prediction": "Malignant / High Risk",
  "cancer_probability": 0.7412,
  "quantum_confidence_pct": 74.1,
  "top_biomarkers": [{"compound": "...", "importance_score": 0.0, "clinical_impact": "Elevated (Risk Driver)"}],
  "biochemical_pathways": {"Lipid_Peroxidation_Aldehydes": 0.0, ...},
  "clinical_summary": "Patient ... "
}
```

`quantum_confidence_pct = prob_pos*100 if prob_pos >= 0.5 else (1-prob_pos)*100`,
rounded to 1 dp — i.e. confidence *in the predicted class*, not in "cancer".

**Input validation:** `400` if `len(sensor_readings) != 64`. The 64-dim contract
is the *synthetic* schema (16 sensors × 4 kinetics). A user posting the real 27
VOC columns gets a 400 — the screening service does not yet accept the real
dataset's feature dimensionality. See
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md).

**Background data for SHAP is the test matrix.** `server.py:87-91` builds
`shap_explainer = QuantumKernelSHAP(predict_fn, background_data=X_te_q, n_samples_background=16)`.
The *background set* is a sample of the **held-out test split** — it defines
$\mathbb{E}[f]$, the model's output on a typical sample. Using test data as the
background distribution leaks test-set information into the explanation, and is a
train/serve inconsistency. Should be the *training* reduced matrix. This is a
small but genuine methodological defect.

---

## 4. Frontend rendering

`frontend/components/BiomarkerWaterfall.tsx` (114 lines):

- **Section 1** — "Top Attributed VOC Metabolites (Reverse Hilbert Mapping)": one
  row per biomarker, with a chip classified as *elevated* when
  `clinical_impact.includes("Elevated") || includes("Risk")`, the
  `importance_score.toFixed(4)`, and a bar of width
  `min(100, max(8, score/maxScore*100))` — the `max(8, …)` floor keeps small bars
  visible.
- **Section 2** — "Biochemical Pathway Dysregulation": a 2-column grid of
  `pathway.replace(/_/g," ")` + percentage + mini bar, rendered only when
  `pathways` is non-empty.
- **Empty state** — "No biomarker attribution computed for this window."

Displayed on `/diagnostics` only when `detailedResult` is set (i.e. after an
explicit `POST /api/v1/predict?deep_explain=true`); the WebSocket-driven
`PredictionCard` path shows no attribution, because the live kennel path is a
RandomForest on 45 kennel features with no chemical attribution layer.

**The kennel path has no explainability.** That is a scope boundary worth stating
in the presentation: biomarker attribution exists for the VOC domain only.
