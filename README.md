# Store Sales — Time Series Forecasting

Forecasting 16 days of daily unit sales for **1,782 store × product-family series** at
Corporación Favorita, Ecuador ([Kaggle playground competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)).

A single global LightGBM, trained as a **direct** (non-recursive) forecaster on `log1p(sales)`,
scores **RMSLE 0.39081** on a held-out fortnight — **24.9% better than the strongest naive
baseline** (0.52063, same-weekday mean of the last 8 weeks).

> The interesting part of this repository is not the score. It is the **measurement record**:
> which changes were tested, which were rejected, and — twice — how an effect that was
> confidently reported at first turned out to be an artefact once confounders were held fixed.

---

## The problem

| | |
|---|---|
| **Target** | `sales` for every `(date, store_nbr, family)` |
| **Scope** | 54 stores × 33 families = **1,782 series** |
| **Horizon** | **16 days**, 2017-08-16 → 2017-08-31 |
| **History** | 2013-01-01 → 2017-08-15 (1,684 trading days) |
| **Metric** | **RMSLE** — RMSE computed on `log1p(sales)` |

RMSLE is what drives every design decision here. The error is **relative**, so a 5-unit series
matters as much as a 5,000-unit one: accuracy on the long tail is where the score is won, not
on `GROCERY I`. The target is modelled as `log1p(sales)` and inverted with `expm1`, clipped at
zero — which makes the training loss *exactly* the competition metric.

## Results

Public leaderboard progression (this competition has no private leaderboard — the public score
covers the full test set):

| Submission | LB | Local holdout | Note |
|---|---|---|---|
| Single LightGBM | 0.55331 | 0.39662 | |
| + 3-seed ensemble | 0.54284 | 0.39472 | transferred ~3.5× its measured value |
| + promotion rework (relative intensity + known-future leads) | 0.46264 | 0.39763 | **local validation said this would *hurt*** |
| + family/store promotion intensity, `min_data_in_leaf` 50 | 0.42093 | 0.39344 | |
| + store traffic (`transactions`) | 0.42400 | — | ❌ reverted, +0.00307 |
| + round cap 1200→2400, patience 50→150 | 0.42074 | **0.39081** | current |

Reference points on the same holdout: all-zeros 4.4195 · last observed day 0.6595 ·
mean of last 16 days 0.5224 · **same-weekday mean of last 8 weeks 0.5206** (the naive bar).

## Approach

- **One global model, not 1,782 local ones.** Pooling lets sparse series borrow strength from
  dense ones; per-family models were measured and *lost* (0.42149 vs 0.40457).
- **Direct, not recursive.** Feeding predictions back for 16 steps compounds error faster than
  `lag_1` repays: recursive pooled scored 0.46879 against 0.40457 for direct.
- **Every lag and rolling window is shifted by ≥ 16 days**, so the whole horizon is forecastable
  without recursion. `onpromotion` is the deliberate exception — it is *given* for the test
  window, so current-day and short-lead values are legal and are used heavily.
- **5-seed ensemble averaged in log space**, with `feature_fraction`/`bagging_fraction` varied
  per seed.
- **The panel is held wide** (dates × series, 1,704 × 1,782) so a lag or rolling window is one
  vectorised operation instead of a groupby over 3M rows.

### Repository layout

```
data/                                        raw competition CSVs (read-only)
notebooks/
  01_exploratory_data_analysis.ipynb         EDA + effect sizing
  02_feature_engineering_and_modelling.ipynb features, LightGBM, ablations
kaggle_store_sales_submission.ipynb          the file uploaded to Kaggle
stakeholder_sales_forecast_report.ipynb      business-facing report (same model, no ablation log)
submissions/                                 generated forecasts
CLAUDE.md                                    full engineering log and working rules
```

---

## What makes this project unusual: the measurement record

### A bug that backtesting structurally could not catch

A pair of features asking *"is there a big promotion anywhere in the 16-day forecast horizon?"*
was built, backtested on 2015 and 2016, and appeared to win cleanly on both years. **It was
never submitted.** A visual check of the forecast chart — not the summary statistics — caught a
catastrophic collapse on the final forecast day.

The cause: near the true end of the data, a 16-day forward window has nothing left to look
forward to. `np.nansum` over an all-missing slice silently returns **0.0**, indistinguishable
from *"confirmed zero promotion coming"* rather than *"unknown"*. Every training row sits years
from the panel's edge, where full lookahead is always available, so the model had never seen
that missing-data pattern and extrapolated badly on it.

Measured in a controlled A/B on the real forecast: the last day's predicted volume collapsed to
**~30% of correct** (196k vs 664k chain-wide units). The 2015/2016 backtests could not have
caught it — those target windows are never near the panel's end, so the failure mode was never
exercised.

**Two lessons, both now standing rules here.** A feature that reaches toward the edge of what
the data provides must be tested *at that edge*. And: render the forecast continuity chart
before calling a rebuild done — `0 errors` and matching aggregate statistics all looked fine,
because a mean masks a single catastrophic day.

### Effect sizes that changed materially when confounders were held fixed

The holiday effect was measured three times and the answer moved by more than an order of
magnitude:

| Pass | Baseline used | National holidays | Why the earlier number was wrong |
|---|---|---|---|
| 1 | ratio of means, pooled | **+19.8%** | series composition + weekday mix |
| 2 | all-time mean per (series, weekday) | **×1.041** | does not control for season |
| 3 | same weekday within ±28 days | **×1.012** | — |

28 of 82 national-holiday occurrences fall in December, the annual peak. Measuring a December
holiday against a series' all-time average partly measures December. Oil moved the same way:
**r = −0.63 raw, r = −0.02 once detrended** — two coincident trends, not a relationship.

On this data a ratio of means is not evidence: the zero mass, series composition and the weekly
cycle all distort it. Effects here are quoted in log space or as quantiles, computed *within*
series and weekday.

### Tested and rejected

Every row below was measured, not assumed. Most are negative results, kept deliberately.

| Change | Verdict |
|---|---|
| Recursive forecasting | ❌ 0.46879 vs 0.40457 direct |
| Per-family models | ❌ helps recursive models, *hurts* direct ones — opposite signs |
| Store traffic (`transactions`) | ❌ +0.00307 on the leaderboard; redundant given store identity |
| Oil price | ❌ r = −0.02 detrended, +0.00 differenced |
| Binary `is_holiday` flag | ❌ pooled ×1.01 — it averages New Year (×0.07) with Puente Navidad (×1.86) |
| Regional holidays | ❌ ≤ 5 observations per series |
| Yearly seasonality (`lag_364`) | ❌ `SCHOOL AND OFFICE SUPPLIES` grew ~12× YoY in the target window |
| Dormant-series hard zeros | ❌ no-op — recent-history features already predict ~0 there |
| Earthquake dummy flag | ❌ **worst** of three treatments; see below |
| Linear trend + Fourier blend | ❌ 0.92 on holdout, worse than naive |
| Two-stage model (P(sale) × magnitude) | ❌ statistical tie (0.39421 vs 0.39428); low-volume deciles got *worse* |
| Back-to-school YoY ramp | ❌ target family degraded 0.702 → 0.806 |
| Regional (Sierra/Costa) promotion intensity | ❌ lost on all three windows; 8 of 9 paired comparisons |
| Prediction bias / multiplier correction | ❌ optimal offset **flips sign** between backtest years |
| XGBoost ensemble | ❌ residual correlation with LightGBM is **0.974–0.987** |
| **3-seed → 5-seed ensembling** | ✅ **the only survivor, ~0.003** |

**A real effect in the data does not imply a feature that helps.** Several of the rejections
above started from a *verified* effect — a +19% month-turn lift, a ×25 back-to-school promotion
surge, r = 0.84 store-day transactions, a genuine Sierra/Costa school-calendar split — and lost
anyway. With `day`/`dow`/`dayofyear`/series identity plus recent-level features already in the
model, explicit encodings mostly compete for `feature_fraction` budget instead of adding
information.

### Measurement rules learned the hard way

1. **The seed-noise floor on this holdout is ~0.0039.** Three fits differing only by random seed
   span that much. A single-seed margin below it is *unmeasured*, not small. Three changes were
   nearly adopted on margins of 0.00067, 0.00004 and 0.00091 before this was quantified.
2. **Compare paired, not pooled.** Run the same seeds under both arms so seed variation cancels
   in the difference. This recovers roughly an order of magnitude of sensitivity, for free.
3. **Never use a random split.** Adjacent days leak. Validation is a contiguous 16-day block.
4. **Match the validation window's calendar position**, not just its length. The correct
   backtest is origin 15 August, target 16–31 August, with stores that go dark excluded.
5. **Test the rule, report what the measurement says.** The dormant-zero rule and the earthquake
   quarantine both looked obviously correct and both measured as no-ops. They are reported as
   no-ops.

### The validation ↔ leaderboard gap

Local holdout said 0.394 while the leaderboard said 0.543 — a 0.15 gap. Four structural
hypotheses were tested and eliminated (wrong fortnight, store closures, misaligned submission,
horizon degradation). What actually explained it emerged from tracking the gap across every
submission: it closed **monotonically, in lockstep with every promotion-related change**, from
0.157 down to 0.027.

`onpromotion` is the one input where the test window differs in *kind* from anything inside the
training data — 2017's promotional ramp into 16–31 August is the largest of any year on record
(promo count ×1.18, against ×1.10 in 2016 and ×0.92 in 2015). A model seeing only a flat
per-series promotion count cannot represent that shape; one seeing it relative to each series'
own norm, and at family, store and chain level, increasingly can. The 2015/2016 backtests
contain no ramp of comparable scale, which is why every step of the closure arrived as a
leaderboard surprise rather than a predicted result.

The remaining 0.027 is **not** a calibration problem — a bias probe (`target ≈ a + b·pred`)
returns `b ≈ 1.00` on all three windows, and the optimal offset flips sign between backtest
years. That explanation is eliminated; what remains is variance or missing information.

---

## Notable findings

- **Where the score comes from.** 95% of LightGBM's gain over naive is "recent level" — lags,
  rolling means, same-weekday history. `rmean_7` alone accounts for 63%, `dow_mean_4` for 15%.
  Holidays, calendar, series identity, promotions, dormancy and volatility split the remaining
  5%, roughly 1% each.
- **Where the error lives.** The bottom three volume deciles carry **46% of all squared error**,
  because RMSLE weights every row equally. The worst family is `SCHOOL AND OFFICE SUPPLIES`
  (0.78) — August is the back-to-school ramp, with only two prior Augusts to learn it from.
  `BOOKS` scores 0.10 because it is almost always zero and the model correctly predicts zero.
- **Ecuador has two school calendars.** The Costa region returns to class in April–May
  (5.56× its annual mean), the Sierra in August–September (5.44×). The forecast window sits
  exactly on the Sierra peak. Promotion intensity entering the window rises ×6.66 in the Sierra
  against ×1.65 on the Costa — a split the national aggregate (×5.12) hides completely. Real,
  quantified, and *still* not a useful feature (see the rejection table).
- **The earthquake contaminates far beyond its own week.** The April 2016 shock lasted 7 days,
  but with a 112-day rolling mean at lag 16, rows as late as **August 2016** carry tainted
  inputs — 135 days of feature contamination from a 7-day event. It is repaired at source in the
  panel rather than flagged with a dummy; the dummy measured as the *worst* of three treatments,
  since it lets trees split on a flag active in 0.74% of rows while leaving the contamination
  untouched.
- **The panel is a complete rectangle** — 54 × 33 × 1,684 dates, zero nulls. Four dates are
  absent from the calendar (25 December, 2013–2016, all stores shut) and must be reinserted as
  zeros before building lags. 31% of `sales` rows are exactly zero, mostly structural; 15% are
  fractional, because products are sold by weight.

---

## Running it

The working interpreter is a global Python 3.12 install — `.venv/` exists but is empty.

```bash
pip install -r requirements.txt

# notebooks resolve DATA by probing ./data and ../data, so either directory works
python -m nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace kaggle_store_sales_submission.ipynb
```

Notebooks are committed **with outputs** and must run top to bottom without error.
`jupyter` is not on `PATH`; invoke it as `python -m nbconvert`. Full training is ~57 minutes
(5 seeds for validation, 5 for the final fit).

Load `train.csv` with explicit dtypes — 78 MB instead of ~500 MB:

```python
pd.read_csv("train.csv", parse_dates=["date"], dtype={
    "store_nbr": "int8", "family": "category",
    "onpromotion": "int32", "sales": "float32"})
```

## Where this could still go

Ranked by expected value, after nine consecutive well-measured rejections:

1. **Per-horizon models** — one per day of the 16, letting near days use short lags the current
   design must forgo. The only remaining change that alters *what information each row may see*.
   A prototype showed −0.0124 at h=1; honest expected gain ~0.005, and it needs a multi-origin
   validation harness to measure correctly.
2. **The loss function** — Tweedie or Poisson on raw sales, rather than RMSE on `log1p`. Attacks
   the intermittent low-volume regime by a different mechanism, and costs one fit to rule out.

The feature set is close to saturated. Changes that give the model **information it did not
previously have** transferred at or far above their measured value; changes that let the same
model fit the same information better transferred at roughly 7% of it.
