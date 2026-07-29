# TrueOdds

**A probability-calibration lab that fits temperature scaling and isotonic regression, live in the browser, against real held-out classifier outputs.**

On the worst-calibrated of four models tested (Gaussian naive Bayes), isotonic regression cuts
expected calibration error from 0.1368 to 0.0116 on 9,769 untouched test cases, a 91.5% reduction,
while temperature scaling's strictly monotone transform leaves rank AUC exactly, bit-for-bit
unchanged at 0.8878. Recalibrating how much a probability can be trusted does not require touching
how well the model ranks cases, and the results table below shows both the wins and the one case
where recalibration makes a proper score slightly worse.

![python self-check](https://img.shields.io/badge/python%20self--check-7%2F7-brightgreen)
![js self-check](https://img.shields.io/badge/js%20self--check-6%2F6-brightgreen)
![models compared](https://img.shields.io/badge/models%20compared-4-blue)
![ECE reduction](https://img.shields.io/badge/ECE%20reduction-up%20to%2091.5%25-orange)
![license](https://img.shields.io/badge/license-MIT-green)

## The problem

Classification accuracy does not say whether a predicted probability can support a threshold,
expected-value calculation, or triage rule. A model can rank cases well and still hand out
probabilities that are systematically too confident or too timid. TrueOdds separates ranking
quality from probability quality: four models are fit offline on UCI Adult, their calibration and
test probabilities are committed as data, and every recalibrator and metric is then computed in
vanilla JavaScript. The result shows what calibration changes, what it leaves alone, and when
post-hoc calibration can make a model worse.

## How it works

- `scripts/build_data.py` downloads UCI Adult once, does a fixed 60/20/20 split (model fit,
  recalibrator fit, untouched test), trains four scikit-learn models, and exports only split labels
  and four-decimal test/calibration probabilities to `data/adult.json`. No raw demographic features
  ship to the browser.
- The page fetches that one same-origin JSON file and does the rest with no backend: reliability
  diagrams, ECE/MCE, Brier with the Murphy decomposition, log loss, and rank AUC, for the raw
  probabilities and for two recalibrators.
- Temperature scaling and isotonic regression (PAVA) are fit live, in the browser, on the
  calibration split only, then applied to the test split. Neither recalibrator, nor the original
  model, ever sees test labels while fitting.
- A false-positive/false-negative cost panel turns the calibrated probabilities into a decision
  threshold and an observed cost per 100 cases.

## Method

- **Reliability diagram:** each point compares the mean issued probability in a bin with its
  observed positive rate. Both equal-width and equal-frequency bins are available, and the count
  behind every point appears in a histogram and accessible table.
- **ECE and MCE:** `ECE = sum_m |B_m| / n * |acc(B_m) - conf(B_m)|`. MCE is the largest absolute
  bin gap. Both depend on the binning rule, bin count, and sample size. The interface exposes these
  choices because legacy binned ECE is a biased estimator and can change model comparisons.
- **Brier score:** `BS = n^-1 sum_i (p_i - y_i)^2`.
- **Murphy decomposition:** `BS = REL - RES + UNC`, where reliability measures forecast-event
  disagreement, resolution measures separation between event-rate groups, and
  `UNC = y_bar(1 - y_bar)`. The browser groups identical issued forecast values, rather than the
  display bins, so the three terms reconstruct Brier exactly within floating-point tolerance (the
  largest gap observed across all twelve model/recalibrator combinations is 7.2e-15).
- **Log loss:** mean binary negative log likelihood after clipping only for numerical safety.
- **AUC:** computed from average ranks with tie handling and no library. Temperature scaling is
  strictly monotone, so AUC is unchanged exactly. Isotonic regression is non-decreasing but creates
  ties, so its empirical AUC can move slightly, as it does here (0.8878 to 0.8876 on naive Bayes).
- **Temperature scaling:** `p_T = sigmoid(logit(p) / T)`. One positive scalar is selected by
  minimizing calibration-split NLL with a bounded ternary search over `T in [0.05, 20]`.
- **Isotonic regression:** a pool-adjacent-violators algorithm sorts calibration probabilities and
  repeatedly merges adjacent blocks whose fitted event rates decrease. The resulting step function
  is applied to test probabilities.
- **Decision threshold:** with false-positive cost `C_FP` and false-negative cost `C_FN`, predict
  positive when `p >= C_FP / (C_FP + C_FN)`. The page reports probability-implied expected cost
  and observed held-out cost per 100 cases.

The implementation follows [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) for
single-parameter temperature scaling,
[Niculescu-Mizil and Caruana (2005)](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf)
for independent calibration data and the flexibility versus overfitting tradeoff of isotonic
regression, and [Murphy (1973)](https://doi.org/10.1175/1520-0450%281973%29012%3C0595%3AANVPOT%3E2.0.CO%3B2)
for the Brier partition. Current practice was checked against the
[scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) and
[Kumar et al. (2021)](https://arxiv.org/abs/2012.08668) on bias in binned calibration-error
estimation.

Research changed two design choices. The Murphy terms use distinct forecast values instead of
reliability-display bins so the stated three-term identity is exact. The AUC copy also distinguishes
strictly monotone temperature scaling from tie-producing isotonic steps instead of claiming that
all calibration preserves ranking exactly.

## Data

[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) contains 48,842 census-derived records and
a binary target for annual income above $50K. Becker and Kohavi published the dataset through the
UCI Machine Learning Repository under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI
[10.24432/C5XW20](https://doi.org/10.24432/C5XW20).

`scripts/build_data.py` downloads the original UCI archive, preprocesses numeric and categorical
features, and fits four fixed-seed scikit-learn models on 29,305 rows (60%):

- logistic regression
- random forest
- gradient boosting
- Gaussian naive Bayes

Calibration probabilities (9,768 rows, 20%) and test probabilities (9,769 rows, 20%) are exported at
four decimals. Only split labels and probabilities are committed; raw demographic features are not
sent to the browser. The artifact is 0.56 MB.

## Results

These are browser-equivalent test metrics with the default 10 equal-width bins. ECE changes when
the user changes the scheme or count. `Temp` and `Iso` were fit only on the independent calibration
split.

| Model | Probabilities | ECE | Brier | Log loss | Rank AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | Raw | 0.0089 | 0.1027 | 0.3212 | 0.9048 |
| Logistic regression | Temp | 0.0101 | 0.1028 | 0.3213 | 0.9048 |
| Logistic regression | Iso | 0.0077 | 0.1029 | 0.3299 | 0.9045 |
| Random forest | Raw | 0.0223 | 0.0953 | 0.3040 | 0.9169 |
| Random forest | Temp | 0.0152 | 0.0945 | 0.3009 | 0.9169 |
| Random forest | Iso | 0.0087 | 0.0943 | 0.3040 | 0.9163 |
| Gradient boosting | Raw | 0.0189 | 0.0931 | 0.2982 | 0.9184 |
| Gradient boosting | Temp | 0.0151 | 0.0930 | 0.2956 | 0.9184 |
| Gradient boosting | Iso | 0.0127 | 0.0929 | 0.3008 | 0.9182 |
| Gaussian naive Bayes | Raw | 0.1368 | 0.1515 | 0.7692 | 0.8878 |
| Gaussian naive Bayes | Temp | 0.1081 | 0.1286 | 0.4156 | 0.8878 |
| Gaussian naive Bayes | Iso | 0.0116 | 0.1122 | 0.3480 | 0.8876 |

The contrast is intentional. Naive Bayes is strongly overconfident and benefits substantially: ECE
falls 20.9% under temperature scaling and 91.5% under isotonic regression, and log loss falls from
0.7692 to 0.3480 under isotonic. Logistic regression is already well calibrated, so either
recalibrator slightly worsens Brier and log loss on new data (ECE still improves under isotonic).
That failure mode is part of the result, not an edge case hidden from it.

## Reproduce

Train the models and rebuild the committed artifact:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_data.py
```

Serve the static site:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080`. Add `?selftest=1` to run the browser assertions and print a pass/fail
summary in the console. The same JavaScript math check can run without a browser:

```bash
node app.js
```

## Security

TrueOdds has no backend, database, model API, LLM, or BYOK mode. The page fetches one committed
same-origin JSON artifact and performs all analysis locally. It does not collect user data, use
storage, load CDN scripts, or send analytics.

## Limitations

- ECE and MCE are bin-dependent estimators. Small bins have high variance, and this project does
  not add confidence intervals or debiased ECE.
- Calibration learned on one population can fail after covariate, label, or concept shift.
- Isotonic regression can overfit a small calibration set and can reduce ranking resolution by
  introducing tied probabilities, which is why its AUC in the results table above moves slightly
  even though it never sees test labels.
- Four-decimal export is sufficient for this demonstration but is not appropriate when decisions
  depend on finer probability differences.
- UCI Adult is a historical benchmark with documented social and demographic limitations. These
  models are a calibration study, not a system for income or employment decisions.

## License

MIT, see [LICENSE](LICENSE).
