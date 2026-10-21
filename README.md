# Harm-Budgeted Probability Updates

Core research code for **Harm-Budgeted Probability Updates for Deployed Risk Models**.

This source-only release contains the allocation algorithms, prediction baselines,
evaluation metrics, predictor-training components, and mechanism experiments used
for the manuscript. Numerical kernels are taken from the executed experiment
sources. Package imports, file interfaces, and command-line entry points have been
adapted for a portable repository. `CODE_PROVENANCE.json` records the original
source hashes and the scope of these adaptations.

## Installation

Python 3.10 or newer is required. The numerical and tree-model experiments require
NumPy, SciPy, and scikit-learn.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. GraphSAGE and the optional CUDA Point backend additionally require a
PyTorch installation suitable for the intended CPU or CUDA environment:

```bash
python -m pip install -e ".[gnn]"
```

The CPU allocator is the default. No GPU is required for frozen-prediction,
finite-source, tree-inference, or dependence experiments.

## Quick functional run

The following example generates artificial inputs and verifies the complete
allocation/evaluation interface. Its outputs are demonstration measurements,
not the manuscript's benchmark results.

```bash
python examples/make_demo.py --output data/demo
python -m harm_budgeted_updates.frozen --input data/demo/frozen --output outputs/demo --config configs/primary.json
python -m harm_budgeted_updates.summarize --input outputs/demo/metrics.csv --output outputs/demo_summary --resamples 1000
python -m unittest discover -s tests -v
```

Allocate into a new output directory. Allocation records are written before
evaluation responses are loaded. The two stages can also be run separately:

```bash
python -m harm_budgeted_updates.frozen --input data/demo/frozen --output outputs/two_stage --stage allocate
python -m harm_budgeted_updates.frozen --input data/demo/frozen --output outputs/two_stage --stage evaluate
```

## Repository contents

| Module | Purpose |
| --- | --- |
| `allocation.py` | Point, Common, UC, cap path, cost-matched logit interpolation, mixtures, and robust comparator refinement |
| `_binary.py`, `_paths.py`, `_robust.py` | Original numerical optimization kernels and deterministic budget calculations |
| `_point_backend.py` | Optional double-precision CUDA dual search with CPU feasibility reconstruction |
| `bayesian.py` | Hard/soft confusion-likelihood Bayesian prediction updates and decision rules |
| `metrics.py`, `queues.py` | NLL, AP, AUROC, Recall@1%, regression, flips, and full-cohort queue outcomes |
| `frozen.py`, `summarize.py` | Probability-only allocation, locked evaluation, and paired graph/draw summaries |
| `training/` | HGB/RF, affine calibration, LP/C&S, GraphSAGE compatibility objectives, and chronological fits |
| `finite_sources.py`, `tree.py`, `known_probability.py` | Finite-source and exact conditional-tree experiments |
| `dependence.py` | Correlated-response experiment with unchanged response marginals |
| `target_paths.py` | Probability/logit target contraction and sorted-capacity UC |
| `configs/` | Primary allowance, original budget sweep, and enlarged-label allowance grid |

## Algorithms and constraints

For deployed probability `b[i]` and proposed probability `p[i]`, the individual
worst-response log-loss increase is

```text
c[i] = max(log(b[i] / p[i]), log((1-b[i]) / (1-p[i])))
C = mean(c)
```

The constraints are `C <= h` and `c[i] <= tau`. The primary operating point is
`h=0.0025`, `tau=0.1`, with interval sensitivity factor `kappa=0.5` and cap-path
level `0.025`. A changed deployment cohort requires a new allocation.

The runner retains these methods:

- `Input`: unchanged deployed probabilities.
- `Native`: the uncapped updated target.
- `Average50`: the unconstrained mean `(b+t)/2`.
- `CapOnly`: the target clipped to the individual probability limits.
- `Common`: a probability-space step toward that capped target.
- `Logit`: a logit-space step toward the same capped target, using the final
  experiment's 58-step scalar search to match Common's attained batch cost.
- `UC`: cost equalization up to each target's individual capacity.
- `Point`: minimum target cross entropy under both constraints.
- `CapPath`: Point with effective cap `min(tau, max(gamma_UC, cap_level))`.
- `RCommon` and `RUC`: interval-based worst-excess-risk refinement from their
  respective feasible comparators.
- Point/Common and Point/UC mixtures with Point weights `0.25`, `0.5`, and `0.75`.
- `Bayes-hard` and `Bayes-soft`, when observed calibration predictions are present.

Native, Average50, and the Bayesian posteriors have unrestricted cost. CapOnly
enforces the individual cap but can exceed the aggregate allowance. Their actual
costs are reported. Bayesian cost and regression include initialization and are
measured from the original deployed `b`.

Real-graph sensitivity intervals default to
`[max(0,t-kappa*min(b,1-b)), min(1,t+kappa*min(b,1-b))]`.
They are sensitivity sets; robust conditional-risk claims require the stated
containment assumption. Supplied `lower` and `upper` arrays replace these sets.
The deterministic regression constraints themselves do not require containment
or independent responses.

## Frozen-prediction inputs

Store one acquisition/predictor or chronological origin/predictor in each NPZ
file. Use a unique filename for every case and arrays loadable with
`allow_pickle=False`.

| Key | Shape | Meaning |
| --- | --- | --- |
| `b`, `t` | `[n]` | Aligned finite probabilities strictly inside `(0,1)` |
| `y` | `[n]` | Evaluation response: `0`, `1`, or `-1` for unknown |
| `ids` | `[n]` | Optional unique identifiers for stable score ties |
| `graph` | scalar string | Graph identifier |
| `draw` | scalar integer | Acquisition ID or chronological origin |
| `reference` | scalar string | Deployed predictor identifier |
| `experiment` | scalar string | Target family; defaults to `primary` |
| `lower`, `upper` | `[n]` | Optional probability-interval endpoints |
| `cal_old`, `cal_new`, `cal_y` | `[m]` | Optional observed calibration predictions and labels for Bayesian baselines |

Evaluation and calibration labels must have disjoint roles. Calibration arrays
are the original observed role, which may also support upstream calibration and
propagation choices. They are not a second independent calibration sample.

The allocation stage does not access `y`. Every supplied budget and method is
allocated before the evaluation stage starts. All arrays are aligned by stable
IDs, and probability hashes are checked again during evaluation.

```bash
python -m harm_budgeted_updates.frozen --input predictions/primary --output outputs/primary --config configs/primary.json
python -m harm_budgeted_updates.frozen --input predictions/primary --output outputs/frontier --config configs/budget_frontier.json
python -m harm_budgeted_updates.frozen --input predictions/large_labels --output outputs/large_labels --config configs/large_update.json
```

The output contains the probability maps, allocation records, and one CSV row per
case, predictor, method, and budget. No operating point is chosen using evaluation
responses.

## Metrics and paired summaries

`metrics.csv` includes NLL, gain over Input, noninterpolated average precision
(AP, the AUPRC summary used in the paper), AUROC, Brier score, ECE, mean regression
`H`, top-one-percent regression `tail1`, negative/positive flips, and Recall@1%.
AP and AUROC aggregate score ties correctly. Recall ranks the highest
`ceil(0.01*n_known)` known-response cases and divides recovered positives by all
known positives in that evaluation cohort.

On Elliptic, NLL/AP/Recall use the known-response subset. `C_deploy` includes all
current transactions, while `C_known` describes the known-response subset.
Columns prefixed with `deploy_queue_` rank the full cohort at one-percent
capacity, preserving unknown outcomes and reporting them separately. These
operational queues have a different ranking population from Recall@1%.

```bash
python -m harm_budgeted_updates.summarize --input outputs/primary/metrics.csv --output outputs/summary --left UC --right Logit --resamples 10000
python -m harm_budgeted_updates.summarize --input outputs/elliptic/metrics.csv --output outputs/elliptic_summary --block-length 3 --left UC --right Point
```

Summaries average predictors within each acquisition and then give graphs equal
weight. All methods and budget points in an acquisition use the same resampled
unit. Elliptic origins use circular blocks of length three by default. The output
includes per-graph means, macro means, paired percentile intervals, and the actual
resampling indices. A macro row requires each included graph, preventing an
unavailable baseline from silently changing the comparison scope. Use `--graphs`
to select an explicitly defined scope, and keep target families in separate
`experiment` groups. An interval crossing zero is inconclusive.

## Data and exact table reproduction

Benchmark datasets, fitted checkpoints, frozen prediction archives, generated
results, manuscript files, and machine-access utilities are excluded from this
code release. The commands accept locally prepared public benchmark arrays.
Preserve the original node ordering, feature construction, edge orientation,
label masks, and label coding when converting data.

The manuscript's primary suite uses Amazon, YelpChi, T-Finance, Twitch-FR,
Twitch-PT, and Twitch-RU with two tree predictors and ten acquisitions per graph.
T-Social is a separate scope. The enlarged-label study uses HGB on Amazon,
YelpChi, and T-Finance with the first five acquisitions. Elliptic uses origins
35 through 49. GraphSAGE target families are evaluated separately on YelpChi,
T-Finance, and Twitch-PT with five acquisitions.

Exact published cells require the original converted data, saved acquisition
masks and predictions, library versions, and resampling indices. The public
frontends preserve the computational procedures and expose their settings;
different library environments or regenerated bootstrap indices can change the last
digits. The artificial example only checks execution.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover budget feasibility, Point's target objective, UC worst-response tails,
full-interval comparator retention, cost-matched logit interpolation, tied-score
ranking metrics, Bayesian probabilities, exact tree enumeration, dependent
responses, unknown-label queues, nested label roles, and exclusion of evaluation
responses from model fitting. GraphSAGE checks run when PyTorch is installed.

The release preparation also compared exported numerical kernels with their
executed originals. Those checks are separate from the small public tests.

## Manuscript attribution

Please cite **Harm-Budgeted Probability Updates for Deployed Risk Models** when
using this implementation. Bayesian update routines implement the hard/soft
confusion-likelihood construction discussed in Träuble et al., *Backward-Compatible
Prediction Updates: A Probabilistic Approach* (NeurIPS 2021). Compatibility
training includes ordinary distillation and focal distillation. Dataset use and
attribution should follow each benchmark's original terms.


<!-- Detailed experiment commands -->

## Training the old and new predictors

The optional training modules contain the computational functions used for the
tree, GraphSAGE, and chronological experiments. Host-specific data loaders and
job submission code have been replaced with a portable NumPy interface.
`CODE_PROVENANCE.json` records the original source hashes and adaptations.
The package contains source code only. Obtain the public benchmark data under
their applicable terms and convert them to the following array schema.

### Graph input

Use one NPZ file with `allow_pickle=False` compatibility:

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | `[n,d]` | Finite node attributes, retaining the intended numeric precision. |
| `y` | `[n]` | Binary response, with `-1` for unknown responses. |
| `edge_index` | `[2,E]` | Integer node indices. Supply both directions for an undirected graph. |
| `ids` | `[n]` | Optional unique original identifiers. |
| `train` | `[n]` | Optional Boolean old-model training mask. |
| `calibrate` | `[n]` | Optional Boolean calibration mask. |
| `acquire` | `[n]` | Optional Boolean additional-label mask. |
| `evaluate` | `[n]` | Optional Boolean evaluation mask. |

Supply all four role masks to reproduce a saved acquisition. The internal names
`select` and `test` are also accepted in place of `acquire` and `evaluate`.
Role masks must be disjoint and refer to known responses. If masks are omitted,
the original v16 label-blind uniform sampler assigns 500 training, 200
calibration, and 50 additional labels. Its dataset name and draw determine the
permutation. Other saved experimental splits should be supplied explicitly.
The graph is transductive: attributes and edges are available for all nodes,
while evaluation responses are erased from the fitting view.

### Graph-feature tree pipeline

```bash
python -m harm_budgeted_updates.training.tree_pipeline \
  --data data/yelp_chi.npz --dataset yelp_chi --draw 0 \
  --target-count 550 --output predictions/tree_550
```

The design matrix concatenates local attributes and their row-normalized
neighbor mean. HGB uses 200 boosting iterations, depth 3, learning rate 0.05,
and squared regularization 1. RF uses 300 trees, square-root feature sampling,
and a minimum leaf size of 2. Each model follows the original regularized
affine calibration, logit averaging, label propagation, and Correct & Smooth
pipeline. LP/C&S interpolation weights are selected from the original grid
using calibration responses only.

The larger-label run retains the deployed predictor and calibration role,
adds 4,450 label-blind sampled evaluation nodes to the original 550-label
target training set, and removes those nodes from evaluation:

```bash
python -m harm_budgeted_updates.training.tree_pipeline \
  --data data/yelp_chi.npz --dataset yelp_chi --draw 0 \
  --target-count 5000 --output predictions/tree_5000
```

This requires enough labeled nodes to leave a nonempty evaluation set.
The output metadata distinguishes `primary` and `large_labels` experiments.

### Compatibility-trained GraphSAGE

Install PyTorch separately for the intended CPU or CUDA environment.

```bash
python -m harm_budgeted_updates.training.graphsage \
  --data data/yelp_chi.npz --dataset yelp_chi --draw 0 \
  --device cuda --output predictions/sage
```

The original two-layer mean GraphSAGE uses 32 hidden units, dropout 0.3, Adam
with learning rate 0.01 and weight decay 0.0005, at most 80 epochs, and
patience 12. Old-training attributes determine standardization. The 200-label
calibration role is split into stopping/calibration/target-selection subsets
of 60/70/70 labels. Six target fits are retained: ordinary cross-entropy,
distillation strengths 0.25/0.5/0.75, and focal-distillation strengths 0.1/1.
The focal-distillation temperature is 100 and its old-correct multiplier is 6.
An optional `--selected-variants` flag also emits the original calibration-
selected distillation families. The default emits the six fixed objectives
reported in the final comparison. Output metadata distinguishes the target
objective, preventing accidental pooling.
`--device cpu` changes execution placement without changing the objective.

### Chronological Elliptic forecasts

The input additionally contains integer `time_step[n]`. By default, `x`
contains exactly the 93 local attributes and excludes the timestamp and
aggregate graph attributes. When a converted feature array retains a leading
timestamp, pass `--feature-start 1 --feature-stop 94` explicitly.

```bash
python -m harm_budgeted_updates.training.temporal \
  --data data/elliptic.npz --output predictions/elliptic \
  --origins 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
```

At origin `s`, the deployed model has label cutoff `s-2` and the new model has
cutoff `s-1`. A cutoff `c` uses training responses through `c-3`, followed by
calibration responses from `c-2` through `c`. Current and future responses are
erased before each fit. The one-step label delay is an explicit experimental
arrival convention, since actual adjudication timestamps are unavailable.
Every current transaction is included in the probability output, including
transactions with unknown responses. Allocation therefore uses the complete
deployment cohort. Response-dependent metrics use only known responses.

### Frozen output

Each generated NPZ contains `b`, `t`, `y`, `ids`, `graph`, `draw`, `reference`,
and `experiment` and can be passed directly to the frozen-prediction runner.
Tree/GraphSAGE outputs also include `cal_old`, `cal_new`, and `cal_y` from
their calibration role. These arrays enable a separate Bayesian baseline
without using evaluation responses. Output records include role and
prediction hashes. New model parameters are fitted only to the designated
training roles, and all allocation algorithms operate subsequently on frozen
probabilities.

Reproducing the exact reported values additionally requires the original
public data conversion, acquisition masks, predictor/library versions, and
the saved frozen predictions. The portable training interface preserves the
core architecture, objectives, calibration, graph propagation, and chronology;
it does not bundle the benchmark data or fitted model artifacts.

## Mechanism experiments and deployment queues

These modules retain the inference, allocation, simulation laws, default
factorial settings, and random-number order used in the paper experiments.
The changes to the original experiment programs are package imports,
configurable input/output directories, and optional smoke-run settings.
Run each command into a new output directory. The programs refuse to replace
an existing protocol record.

## Exact conditional-tree experiment

```bash
python -m harm_budgeted_updates.tree --output results/tree
```

The default experiment uses 30 independent 63-vertex binary-tree realizations,
three interaction strengths, three observed-response densities, three local
region radii, and three omitted-interaction multipliers. It compares exact
full-tree conditional probabilities with region-based intervals and evaluates
Common, UC, Point, mixtures, and comparator-based robust allocation.

For a short functional check:

```bash
python -m harm_budgeted_updates.tree --output results/tree_smoke --smoke
```

This writes a single small factorial condition. It is a functional check and
does not reproduce the paper's aggregate estimates.

## Finite-source and known-probability experiment

```bash
python -m harm_budgeted_updates.finite_sources --output results/finite
python -m harm_budgeted_updates.known_probability --finite-input results/finite --tree-input results/tree/native --output results/known_probability
```

The finite input generator uses 16 source groups and 100 binomial trials per
group. Each group supplies a target to 16 deployment cases. Rarity, odds bias,
interval width, and the fraction of biased rare groups vary over the original
factorial design. Clopper-Pearson intervals depend on finite source counts,
and true response probabilities are reserved for evaluation diagnostics.

The known-probability program combines these finite inputs with the exact
conditional-tree inputs. It computes interval containment, conditional risk,
expected mean harm, joint-response top-one-percent harm, the robust value,
and both symmetric and direction-sensitive interval-error terms. It uses
512 response vectors per conditional law by default. Aligned, shuffled, and
reversed interval-width placements explicitly use the known probabilities
as diagnostic controls. They are not deployable uncertainty estimators.

Use `--draws` consistently for finite generation and known-probability
evaluation. Reducing `--draws` changes the generated finite experiment and
does not select the same rarity-specific cases as the complete 20-draw run.
The `--joint-repetitions` option changes the Monte Carlo evaluation size.
The default arguments reproduce the original design.

## Dependent-response experiment

```bash
python -m harm_budgeted_updates.dependence --output results/dependence
python -m harm_budgeted_updates.dependence --output results/dependence_smoke --smoke
```

The default experiment evaluates 16,384 joint response vectors for each of
six component sizes and five within-component dependence parameters. All
response marginals, probability maps, and deterministic budgets remain
unchanged across dependence settings. Small cases and limiting component
laws are also enumerated exactly. The comparison concentration bound assumes
independent responses and is reported as an assumption-based reference.

## Full-cohort review queues

```bash
python -m harm_budgeted_updates.queues --input predictions.npz --output results/queues.json --fractions 0.005 0.01 0.02
```

The input archive contains aligned one-dimensional arrays:

- `b`: deployed probabilities, strictly between zero and one.
- `p`: proposed probabilities, strictly between zero and one.
- `y`: responses encoded as 0 or 1, with any negative value meaning unknown.
- `ids`: optional unique case identifiers used to break score ties.

Queue capacity is the ceiling of the fraction times the full deployment
cohort size. Unknown responses stay in the ranked cohort and are counted
separately. Known-positive recall uses all known positives in that cohort
as its denominator. It does not treat unknown responses as negatives.
The output includes lost and entering known positives, queue composition,
known-positive recall, and departure rates by the reference logit margin.
When all responses are known, known-positive recall is ordinary Recall@k.

The queue helper evaluates supplied probability maps. It does not fit a model,
estimate uncertainty, choose an operating point, or read labels during
allocation.
