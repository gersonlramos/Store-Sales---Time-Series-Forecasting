# Store Sales — Time Series Forecasting

I forecast 16 days of daily unit sales for **1,782 store × product-family series** at
Corporación Favorita, Ecuador ([Kaggle playground competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)).

My global LightGBM models score **RMSLE 0.38469** on a held-out fortnight — **26% better than
the strongest naive baseline** I could compare against (0.52063, same-weekday mean of the last
8 weeks) — and **0.38924 on the public leaderboard**. Four **direct** per-horizon models supply
70% of the forecast (35%/90%/50% for three families where I found my own leave-one-out-validated
weight); a **recursive** per-family model supplies the rest.

> The interesting part of this repository is not the score. It's the **measurement record** I
> kept as I went: which changes I tested, which I rejected, and — twice — how an effect I
> confidently reported at first turned out to be an artefact once I held the confounders fixed.

**Current best: 0.38924.** A closing round I thought was final turned up one more
leave-one-out-validated refinement (a per-family dormant-zero threshold for BABY CARE) that
landed inside my own established Kaggle-vs-local noise band — not a confirmed win, but not
contradicted either. See [development/CHANGELOG.md](development/CHANGELOG.md) for the full
version history.

---

## The problem

| | |
|---|---|
| **Target** | `sales` for every `(date, store_nbr, family)` |
| **Scope** | 54 stores × 33 families = **1,782 series** |
| **Horizon** | **16 days**, 2017-08-16 → 2017-08-31 |
| **History** | 2013-01-01 → 2017-08-15 (1,684 trading days) |
| **Metric** | **RMSLE** — RMSE computed on `log1p(sales)` |

RMSLE is what drives every design decision I made here. The error is **relative**, so a 5-unit
series matters as much as a 5,000-unit one: I win the score on the long tail, not on
`GROCERY I`. I model the target as `log1p(sales)` and invert with `expm1`, clipped at zero —
which makes my training loss *exactly* the competition metric.

## Results

My public leaderboard progression (this competition has no private leaderboard — the public
score covers the full test set):

| Submission | LB | Local holdout | Note |
|---|---|---|---|
| Single LightGBM | 0.55331 | 0.39662 | |
| + 3-seed ensemble | 0.54284 | 0.39472 | transferred ~3.5× its measured value |
| + promotion rework (relative intensity + known-future leads) | 0.46264 | 0.39763 | **local validation said this would *hurt*** |
| + family/store promotion intensity, `min_data_in_leaf` 50 | 0.42093 | 0.39344 | |
| + store traffic (`transactions`) | 0.42400 | — | ❌ reverted, +0.00307 |
| + round cap 1200→2400, patience 50→150 | 0.42074 | 0.39081 | single-model reference |
| + oil price (level + moving averages) | 0.42157 | ambiguous | ❌ reverted, +0.00083 |
| TiDE (deep learning) + LightGBM blend | 0.42113 | 0.39012 | ❌ my blend sweep chose 100% LightGBM — see below |
| + per-horizon models | 0.41113 | 0.38991 | transferred **4.4×** its measured value |
| + dormant-series hard zero | 0.39586 | ≈0.3859 | transferred **5.6×**; I found it by diffing submissions, see below |
| + recursive per-family blend (70/30) | 0.39079 | 0.38469 | the first blend partner I found that was both decorrelated and comparable |
| + mixed L2 + Huber objective in the direct half | 0.39320 | 0.39139 | ❌ reverted, +0.00241 — a paired 5/5-seed local *win* that reversed sign on the real window |
| + `rmean_3` in the near-horizon buckets | 0.39094 | 0.38494 | ⚠️ not adopted, +0.00015 — flat, inside environment noise, despite a 5.5σ local win |
| + family-gated blend weight (SCHOOL/HOME APPLIANCES/AUTOMOTIVE) | 0.38930 | 0.38469 | leave-one-out validated across three independent windows, zero sign reversals; transferred at 0.78× |
| **+ BABY CARE per-family dormant threshold (90d vs. global 365d)** | **0.38924** | 0.38469 | **current** — ⚠️ −0.00006, inside my own ~0.0004 Kaggle-vs-local noise band; not a confirmed win, but no reversal either |

Reference points on the same holdout: all-zeros 4.4195 · last observed day 0.6595 ·
mean of last 16 days 0.5224 · **same-weekday mean of last 8 weeks 0.5206** (my naive bar).

### Two forecasters that fail in opposite directions

My final model averages a **direct** forecaster with a **recursive** one, and the reason it
works is mechanical rather than statistical.

My direct model obeys a hard constraint: a day 16 steps out may only use sales at least 16
days old — observed, but stale. My recursive model predicts one day at a time and feeds each
prediction back, so it has *yesterday's* value at every horizon — predicted rather than
observed, and therefore carrying compounding error. Each wins where the other is weak:

| Forecast day | Direct | Recursive |
|---|---|---|
| 1 | **0.38005** | 0.38488 |
| 2 | **0.36759** | 0.38063 |
| 8 | 0.39160 | **0.38629** |
| 16 | 0.44508 | **0.41187** |

I tried six earlier blend candidates and all six failed, and I found the two failure modes
turned out to be separate tests a partner must pass **simultaneously**. XGBoost, lag-depth
ensembles and training-window ensembles were strong but correlated at 0.97–0.99 with my direct
model — indistinguishable from the 0.972–0.979 that two copies of the same model differing only
by random seed produce. A neural network (TiDE), a linear/Fourier model and a Ridge on the same
features all decorrelated properly but sat 0.07–0.14 behind in quality, and a blend sweep gives
a weak partner zero weight no matter how different it is.

The recursive model was the first to pass both tests: residual correlation **0.917** — lower
than my own seed-to-seed noise — while scoring within 0.01 of the direct half.

That the recursive half *degrades* with horizon (0.385 near, 0.41–0.43 far) doubles as my
leakage audit. A model somehow seeing the future would not degrade at all; I assert the
degradation in code rather than just trusting it.

### My largest single gain came from comparing outputs, not from a hypothesis

I was stuck 0.032 behind the best public notebooks, and every hypothesis-driven experiment I
ran that day returned a null result. What actually worked was **comparing submission files row
by row** — mine against the best public one. The test labels are hidden, so I couldn't
decompose either model's error directly; but I knew the other model's aggregate score was
better, so wherever the two forecasts differed *systematically*, the sign of the difference
pointed at my own error.

One difference dominated: **on 2,032 rows the public model predicts exactly 0.0, and mine did
not** — I predicted a mean of 0.567 units there, and as much as 37.9.

The cause is structural. A pooled model shares parameters across all 1,782 series, so it
*cannot* emit exact zero however clear the evidence; a store–category pair that hasn't sold a
unit in over a year still gets a small positive forecast out of it. Under RMSLE, where every row
counts equally regardless of volume, those rows were **3.6% of my forecast, 0.01% of the units,
and roughly 75% of my entire gap** to the best public score.

Zeroing any series with no sales in the preceding 365 days gained me **−0.01527** on the
leaderboard against **−0.00274** I'd measured locally — a 5.6× transfer, the largest in the
project. I set the threshold deliberately conservatively: it sits on a plateau (330–547 days)
where no flagged series had actually sold, and tightening it to 300 days *reverses the sign*,
because mis-zeroing a row that sells 3 units costs about seven times what correctly zeroing a
dead one saves.

**The finding was already half-written in my own notes.** I'd recorded this rule as a no-op —
but I'd measured it on a *naive baseline*, where recent-history methods already predict ~0 on
dead series, with an explicit caveat I'd left myself that it might still pay off on a pooled
model that leaks positive predictions. The caveat was correct and sat unexamined for months. I
learned from this: a null result carrying an untested condition is a live lead, not a closed
one.

### The change that mattered most before that, and why I nearly didn't try it

I'd shifted every feature by **at least 16 days**, because the forecast has to reach 16 days
past the last observation. That's correct for the final forecast day and needlessly
conservative for the first, which is one day out and could legitimately use yesterday's sales.

Splitting into four models — each using the freshest lag its horizon range legally allows —
gained me **−0.00961** on the leaderboard, my second-largest improvement in the project.

I'd actually **tested and rejected** this idea once already. My earlier attempt measured a
+0.0690 *loss* at horizon 16 and I wrote it off. That number came from early-stopping each model
against a single day of data (1,782 rows), so my h=16 model halted at 100 trees. At horizon 16
both designs have identical feature staleness and *must* tie — which, once I fixed the harness,
they now do exactly, to eleven decimal places. My rejection was an artefact of the measurement,
not a property of the method.

| Horizon | model | control | per-horizon | Δ |
|---|---|---|---|---|
| 1 | `lag ≥ 1` | 0.38888 | 0.37927 | −0.00961 |
| 2 | `lag ≥ 2` | 0.38450 | 0.36798 | **−0.01651** |
| 3–4 | `lag ≥ 4` | 0.38031 | 0.37525 | −0.00506 |
| 5–16 | `lag ≥ 16` | 0.39492 | 0.39492 | **0.00000** ← sanity check, not a result |

The last row is the harness proving itself to me: that bucket *is* the control model on those
rows, so any value other than zero would mean I'd wired the assembly wrong. I assert it in code.

## Approach

- **One global model, not 1,782 local ones.** Pooling lets sparse series borrow strength from
  dense ones; I measured per-family models and they *lost* (0.42149 vs 0.40457).
- **Direct, not recursive.** Feeding predictions back for 16 steps compounds error faster than
  `lag_1` repays: recursive pooled scored 0.46879 against 0.40457 for direct, when I tested it.
- **Every lag and rolling window is shifted by at least as many days as the horizon it serves.**
  A target date `d` at horizon `h` has forecast origin `d − h`, so feature `lag_k` is legal iff
  `k ≥ h`. I split the horizon across four models at 1 / 2 / 3–4 / 5–16, each taking the
  freshest data its range allows. I exempt `onpromotion` entirely — it's *given* for the test
  window, so current-day and short-lead values are legal at any horizon, and I use them heavily.
- **5-seed ensemble averaged in log space**, with `feature_fraction`/`bagging_fraction` varied
  per seed.
- **I hold the panel wide** (dates × series, 1,704 × 1,782) so a lag or rolling window is one
  vectorised operation instead of a groupby over 3M rows.
- **A recursive per-family model supplies 30% of my forecast for most categories** (65%/10%/50%
  for three that consistently behave differently — see below), averaged in log space. It's the
  only blend partner I found that's both decorrelated from the direct model (residual correlation
  0.917) and comparable to it in quality.
- **I force structurally dead series to exactly zero** as a post-processing step — a pooled
  model can't represent exact zero, so this fixes what the architecture can't express rather
  than what the features fail to say.

### Repository layout

```
data/                                        raw competition CSVs (not in the repo — see below)
notebooks/
  01_exploratory_data_analysis.ipynb         EDA + effect sizing
  02_feature_engineering_and_modelling.ipynb features, LightGBM, ablations
documentation/
  FEATURE_DICTIONARY.md                        all 60 model inputs: definition, source, legality
final_model/                                 the model to look at — production, LB 0.38924
  kaggle_baby_care_dormant_submission.ipynb    family-gated blend + a per-family dormant threshold
  stakeholder_sales_forecast_report.ipynb      the same model, written up first-person for a general audience
development/                                 the path here — every notebook that was once "the best," in order
  CHANGELOG.md                                 version history — every shipped change and why
  kaggle_naive_control.ipynb                   naive seasonal baseline, submitted for calibration (LB 0.52063)
  kaggle_store_sales_submission.ipynb          single-model reference (LB 0.42074)
  kaggle_horizon_submission.ipynb              the direct half on its own (LB 0.39586)
  kaggle_recursive_blend_submission.ipynb      the uniform-weight blend the family-gate model improves on (LB 0.39079)
  kaggle_family_gate_blend_submission.ipynb    direct + recursive blend, family-gated weight (LB 0.38930)
  kaggle_tide_lgbm_ensemble.ipynb              deep-learning ensemble experiment (null result)
submissions/                                 generated forecasts
CLAUDE.md                                    my full engineering log and working rules
```

---

## What makes this project unusual: the measurement record

### A bug that backtesting structurally could not catch

I built a pair of features asking *"is there a big promotion anywhere in the 16-day forecast
horizon?"*, backtested them on 2015 and 2016, and they appeared to win cleanly on both years.
**I never submitted them.** A visual check of the forecast chart — not the summary statistics —
caught a catastrophic collapse on the final forecast day.

The cause: near the true end of the data, a 16-day forward window has nothing left to look
forward to. `np.nansum` over an all-missing slice silently returns **0.0**, indistinguishable
from *"confirmed zero promotion coming"* rather than *"unknown."* Every training row sits years
from the panel's edge, where full lookahead is always available, so my model had never seen
that missing-data pattern and extrapolated badly on it.

I measured it in a controlled A/B on the real forecast: the last day's predicted volume
collapsed to **~30% of correct** (196k vs 664k chain-wide units). My 2015/2016 backtests could
never have caught it — those target windows are never near the panel's end, so they never
exercised the failure mode.

**Two lessons, both standing rules for me now.** A feature that reaches toward the edge of what
the data provides has to be tested *at that edge*. And: render the forecast continuity chart
before I call a rebuild done — `0 errors` and matching aggregate statistics all looked fine to
me, because a mean masks a single catastrophic day.

### Effect sizes that changed materially when I held confounders fixed

I measured the holiday effect three times, and the answer moved by more than an order of
magnitude:

| Pass | Baseline used | National holidays | Why my earlier number was wrong |
|---|---|---|---|
| 1 | ratio of means, pooled | **+19.8%** | series composition + weekday mix |
| 2 | all-time mean per (series, weekday) | **×1.041** | didn't control for season |
| 3 | same weekday within ±28 days | **×1.012** | — |

28 of 82 national-holiday occurrences fall in December, the annual peak. Measuring a December
holiday against a series' all-time average partly measures December. Oil moved the same way
for me: **r = −0.63 raw, r = −0.02 once detrended** — two coincident trends, not a relationship.

On this data I've learned a ratio of means is not evidence: the zero mass, series composition
and the weekly cycle all distort it. I now quote effects in log space or as quantiles, computed
*within* series and weekday.

### Tested and rejected

Every row below I actually measured, not assumed. Most are negative results, which I kept
deliberately.

| Change | Verdict |
|---|---|
| Recursive forecasting | ❌ 0.46879 vs 0.40457 direct |
| Per-family models | ❌ helps recursive models, *hurts* direct ones — opposite signs |
| Store traffic (`transactions`) | ❌ +0.00307 on the leaderboard; redundant given store identity |
| Oil price | ❌ r = −0.02 detrended, +0.00 differenced |
| Binary `is_holiday` flag | ❌ pooled ×1.01 — it averages New Year (×0.07) with Puente Navidad (×1.86) |
| Regional holidays | ❌ ≤ 5 observations per series |
| Yearly seasonality (`lag_364`) | ❌ `SCHOOL AND OFFICE SUPPLIES` grew ~12× YoY in the target window |
| Dormant-series hard zeros (first pass) | ❌ no-op — recent-history features already predict ~0 there |
| Earthquake dummy flag | ❌ **worst** of three treatments I tried; see below |
| Linear trend + Fourier blend | ❌ 0.92 on holdout, worse than naive |
| Two-stage model (P(sale) × magnitude) | ❌ statistical tie (0.39421 vs 0.39428); low-volume deciles got *worse* |
| Back-to-school YoY ramp | ❌ target family degraded 0.702 → 0.806 |
| Regional (Sierra/Costa) promotion intensity | ❌ lost on all three windows; 8 of 9 paired comparisons |
| Prediction bias / multiplier correction | ❌ optimal offset **flips sign** between my backtest years |
| XGBoost ensemble | ❌ residual correlation with LightGBM is **0.974–0.987** |
| **3-seed → 5-seed ensembling** | ✅ **the only survivor, ~0.003** |

**I learned that a real effect in the data does not imply a feature that helps.** Several of the
rejections above started from an effect I'd *verified* — a +19% month-turn lift, a ×25
back-to-school promotion surge, r = 0.84 store-day transactions, a genuine Sierra/Costa
school-calendar split — and I still lost with them. With `day`/`dow`/`dayofyear`/series identity
plus recent-level features already in the model, my explicit encodings mostly competed for
`feature_fraction` budget instead of adding information.

### Measurement rules I learned the hard way

1. **The seed-noise floor on this holdout is ~0.0039.** Three fits differing only by random seed
   span that much. I learned a single-seed margin below it is *unmeasured*, not small — I nearly
   adopted three changes on margins of 0.00067, 0.00004 and 0.00091 before I quantified this.
2. **I compare paired, not pooled.** I run the same seeds under both arms so seed variation
   cancels in the difference. This recovers roughly an order of magnitude of sensitivity, for
   free.
3. **I never use a random split.** Adjacent days leak. My validation is a contiguous 16-day
   block.
4. **I match the validation window's calendar position**, not just its length. My correct
   backtest is origin 15 August, target 16–31 August, with stores that go dark excluded.
5. **I test the rule and report what the measurement says.** The dormant-zero rule and the
   earthquake quarantine both looked obviously correct to me and both measured as no-ops. I
   report them as no-ops.

### The validation ↔ leaderboard gap

My local holdout said 0.394 while the leaderboard said 0.543 — a 0.15 gap. I tested and
eliminated four structural hypotheses (wrong fortnight, store closures, misaligned submission,
horizon degradation). What actually explained it emerged when I tracked the gap across every
submission: it closed **monotonically, in lockstep with every promotion-related change**, from
0.157 down to 0.027.

`onpromotion` turned out to be the one input where the test window differs in *kind* from
anything inside the training data — 2017's promotional ramp into 16–31 August is the largest of
any year on record (promo count ×1.18, against ×1.10 in 2016 and ×0.92 in 2015). A model seeing
only a flat per-series promotion count can't represent that shape; one seeing it relative to
each series' own norm, and at family, store and chain level, increasingly can. My 2015/2016
backtests contain no ramp of comparable scale, which is why every step of the closure arrived as
a leaderboard surprise to me rather than a predicted result.

The remaining 0.027 is **not** a calibration problem — I ran a bias probe (`target ≈ a + b·pred`)
and got `b ≈ 1.00` on all three windows, and the optimal offset flips sign between my backtest
years. I've eliminated that explanation; what remains is variance or missing information.

---

## Notable findings

- **Where my score comes from.** 95% of LightGBM's gain over naive is "recent level" — lags,
  rolling means, same-weekday history. `rmean_7` alone accounts for 63%, `dow_mean_4` for 15%.
  Holidays, calendar, series identity, promotions, dormancy and volatility split the remaining
  5% between them, roughly 1% each.
- **Where my error lives.** The bottom three volume deciles carry **46% of all squared error**,
  because RMSLE weights every row equally. My worst family is `SCHOOL AND OFFICE SUPPLIES`
  (0.78) — August is the back-to-school ramp, and I only have two prior Augusts to learn it
  from. `BOOKS` scores 0.10 because it's almost always zero and my model correctly predicts
  zero.
- **Ecuador has two school calendars.** The Costa region returns to class in April–May
  (5.56× its annual mean), the Sierra in August–September (5.44×). My forecast window sits
  exactly on the Sierra peak. Promotion intensity entering the window rises ×6.66 in the Sierra
  against ×1.65 on the Costa — a split the national aggregate (×5.12) hides completely. Real,
  quantified, and I still couldn't turn it into a useful feature (see the rejection table).
- **The earthquake contaminates far beyond its own week.** The April 2016 shock lasted 7 days,
  but with a 112-day rolling mean at lag 16, rows as late as **August 2016** carry tainted
  inputs in my panel — 135 days of feature contamination from a 7-day event. I repair it at
  source in the panel rather than flagging it with a dummy; the dummy measured as the *worst*
  of three treatments I tried, since it lets trees split on a flag active in 0.74% of rows while
  leaving the contamination untouched.
- **The panel is a complete rectangle** — 54 × 33 × 1,684 dates, zero nulls. Four dates are
  absent from the calendar (25 December, 2013–2016, all stores shut), and I reinsert them as
  zeros before building lags. 31% of `sales` rows are exactly zero, mostly structural; 15% are
  fractional, because products are sold by weight.

---

## Running it

I use a global Python 3.12 install — `.venv/` exists but is empty.

**I don't keep the competition data in this repository** — `train.csv` alone is 116 MB, over
GitHub's per-file limit. Either download the CSVs into `./data`
([competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)),
or let the notebooks fetch them themselves: each one probes `./data` and `../data` first, then
falls back to `kagglehub.competition_download()`, so they also run unmodified on Kaggle.

```bash
pip install -r requirements.txt

python -m nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace final_model/kaggle_baby_care_dormant_submission.ipynb
```

I commit notebooks **with outputs**, and I keep them running top to bottom without error.
`jupyter` isn't on `PATH` in my setup; I invoke it as `python -m nbconvert`. Full training takes
me ~57 minutes (5 seeds for validation, 5 for the final fit).

I load `train.csv` with explicit dtypes — 78 MB instead of ~500 MB:

```python
pd.read_csv("train.csv", parse_dates=["date"], dtype={
    "store_nbr": "int8", "family": "category",
    "onpromotion": "int32", "sales": "float32"})
```

## Where this could still go

Three items that used to sit here I've since resolved, which I think is worth stating plainly
rather than quietly replacing: **I built and shipped per-horizon models** (−0.00961 on the
leaderboard, transferring at 4.4×), **I closed the loss function question in both directions** —
Tweedie and Poisson lost decisively on local measurement, and a mixed L2+Huber objective I
submitted lost on the leaderboard too — and **I found a second model good enough to earn real
weight, then refined it**: a recursive per-family model, blended uniformly at first (−0.00507,
1.28×) and then given a leave-one-out-validated per-family weight for the three families where
it consistently helps most (−0.00149, 0.78×). That refinement is my production model now.

What my record actually says about where value remains:

| Kind of change | Local → leaderboard | Attempts |
|---|---|---|
| Information the model could not previously see | **0×–5.6×, once with the sign reversed in my favour** | promotion rework, per-horizon models, `rmean_3` |
| A structural inability of the architecture | **5.6×** | dormant-series hard zero |
| A genuinely complementary second model | **0.78×–1.28×, zero reversals in two attempts** | recursive per-family blend, its family-gated refinement |
| Fitting the same information better | **~7%, and once with the sign reversed against me** | round cap, mixed objective |

I don't rate the four categories equally. **Model complementarity is the most reliable category
in my project now** — two attempts, two confirmed wins, transfer ratios 0.78×–1.28×, never
reversed — because I validated both the same disciplined way: paired across seeds or windows the
change was never fit on, not read off the window being reported. Fit optimisation is the
opposite story for me: two clean, paired, multi-seed local wins and two disappointing
deliveries, one an outright sign reversal (Huber). The information category sits between them:
two of three attempts over-transferred by 4×–6× (or more), but the third (`rmean_3`) landed flat
despite a 5.5σ local result — a reminder to myself that a large effect measured on a diluted
slice of the data can shrink back to noise once diluted again at submission time.

So my honest ranking is short:

1. **Ask whether an existing model is genuinely better on some slice, and validate any weight
   change leave-one-out across independent windows before trusting it.** This is what worked
   twice in a row for me — first at the whole-model level (recursive vs. direct), then at the
   family level. The discipline, not the idea, is what made it reliable: every number I report
   for the family-gated weight was computed on a window that weight had never seen.
2. **Anything that adds information about the test window the model can't currently reach.**
   Sixteen feature attempts have failed for me here, all for the same diagnosable reason — trees
   reconstruct the pattern from existing features, so my explicit encoding only competes for
   `feature_fraction` budget. My successes came from *removing a constraint* (lag staleness)
   rather than adding a column, and even those I still need to measure on the actual slice they
   serve, not diluted across the whole window, to know whether they're real before I ship them.
3. **Asking what the architecture can't represent at all.** This question found the largest
   transfer in my project, and I found it by comparing submission CSVs row by row rather than
   by hypothesis.

Not worth further effort, on this evidence: hyperparameters, ensemble size (I've saturated it at
~6 models), objectives, calibration, and any diversity axis that turns out to correlate at 0.97
with a reseed — which so far is all of them except the recursive model.
