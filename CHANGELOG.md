# Version History

How the model went from **0.55331** to **0.38930** on the public leaderboard — nine shipped
changes, in the order they happened, each with the reasoning that justified it at the time.
Rejected attempts are noted inline where they explain why the path bends the way it does; the
full account of every rejection (roughly 40, each with a measured reason) lives in the
[README's measurement record](README.md#what-makes-this-project-unusual-the-measurement-record)
and this repository's private engineering log.

Every score below is the **public leaderboard** — this competition has no private leaderboard,
so the public score is the only real-world number that exists. "Local" is the held-out fortnight
inside `train.csv`, used to decide what to submit before spending a submission on it.

---

## v9 — Family-gated blend weight — **0.38930** (current)
**2026-08-22 · −0.00149 from v8**

Three product categories — `SCHOOL AND OFFICE SUPPLIES`, `HOME APPLIANCES`, `AUTOMOTIVE` — use a
different direct/recursive split than the uniform 70/30 every other category gets (35/65,
90/10, and 50/50 respectively). Found by optimising the split independently on three separate
historical periods and keeping only categories where the same split helped in **all three**,
never picked by looking at the window it would be scored on. Local estimate: −0.00191, blind to
the real leaderboard window. Transferred at 0.78× — smaller than v8's own transfer ratio, but
the same sign, and the third result in a row from this validation discipline (leave-one-out
across independent windows) with zero reversals.

*Rejected the same week, tested with the same rigor:* a mixed L2+Huber training objective
(measured −0.00197 locally, 5/5 seeds; **reversed to +0.00241 on the real leaderboard**) and
`rmean_3` — a third rolling-mean window for the near-horizon models (measured −0.00244, 5.5σ;
landed flat, +0.00015, on the real window). Both were validated on a *single* window, unlike the
family-gated weight's three — the likely reason one reversed, one went flat, and this one held.

## v8 — Recursive per-family blend — 0.39079
**2026-08-20/21 · −0.00507 from v7**

The first "second model" — of seven candidates tried across three weeks — that was both
genuinely different from the tree-based direct model (residual correlation 0.917, versus
0.97–0.99 for every other tree-based variant tried) *and* good enough on its own to earn real
blend weight (RMSLE 0.399 solo, versus a 0.10+ gap for every neural or linear alternative
tried). Trained one small model per product category, predicting one day at a time and feeding
each prediction back in — the opposite failure mode from the direct model: fresh but
increasingly speculative, where direct is reliable but stale. Blended 70/30 in log space.

## v7 — Dormant-series hard zero — 0.39586
**2026-08-19 · −0.01527 from v6, the largest single transfer in the project (5.6× the local
measurement)**

Not a new feature — a post-processing rule. A pooled model sharing parameters across 1,782
series structurally cannot output an exact zero, so any store/category combination with no
sales in the trailing year still received a small positive forecast. Found by comparing this
project's submission against a leading public one row by row: one difference — 2,032 rows where
the public blend predicted exactly zero and this model didn't — explained most of the remaining
gap between them.

## v6 — Per-horizon models — 0.41113
**2026-08-19 · −0.00961 from v5, transferred at 4.4×**

Four models instead of one, each using the freshest sales data its forecast day can legally
see. A single model predicting all 16 days had to obey the horizon-16 staleness rule even on
day 1, when far fresher data was available and legal. No new columns — the same features, just
without a staleness constraint the near-term forecasts never needed.

## v5 — Round cap 1200→2400, patience 50→150 — 0.42074
**~2026-08-18 · −0.00019 from v4**

The model had been stopping before convergence: at the old patience setting it halted around
900 trees; it now stops on its own near 1,725. A clean, paired, 3/3-seed local win of −0.00263
— but it transferred at only ~7% of that to the real leaderboard, the first sign that
hyperparameter-tuning changes behave very differently from changes that give the model new
information.

## v4 — Family/store promotion intensity + tuning — 0.42093
**~2026-08-17/18 · −0.04171 from v3**

Extended the relative-promotion idea from v3 to the family and store scope — not just "is this
series promoting more than usual" but "is this whole product category, or this whole store,
promoting more than usual right now" — and lowered `min_data_in_leaf` from 200 to 50 (a real,
monotonic local win independent of the promotion features).

## v3 — Promotion rework: relative intensity + known-future leads — 0.46264
**~2026-08-17 · −0.0802 from v2, the largest single leaderboard gain in the project**

The single biggest result here, and the one hardest to trust in advance. `onpromotion` had
been used as a raw count; rebuilt as *intensity relative to each series' own 112-day norm*,
plus 1/2/3/7-day-ahead leads (legal because `test.csv` gives promotion status for the entire
forecast window, unlike sales). Locally this looked like a **wash or a loss** — the 2015/2016
backtests predicted +0.0016 to +0.0029 (worse). 2017's actual promotional ramp into the test
window turned out to be the largest on record, a pattern no backtest year contained at
comparable scale. The gap between local and leaderboard scores closed by roughly 80% across this
change and the two that followed it — the clearest evidence in the project that local
validation reliably catches regressions but can badly underestimate a genuine win tied to
information no backtest year happens to contain.

*A closely related feature — a promo signal reaching 16 days into the forecast horizon — was
built, backtested, and looked like a clean win before this one shipped. It was never
submitted*: a visual check of the forecast chart (not the summary statistics) caught a
last-day volume collapse to ~30% of correct, caused by `np.nansum` over an all-missing slice at
the panel's edge silently returning 0.0 instead of "unknown." No backtest year could have shown
this — every backtest sits years from the panel's true edge. The chart is now checked before
every model change of this kind, not just this one.

## v2 — 3-seed ensemble — 0.54284
**~2026-08-16 · −0.01047 from v1, transferred at ~3.5× its local measurement**

Averaging three LightGBM fits differing only by random seed. A small, well-understood variance
reduction — notable mainly because the transfer ratio here first hinted that changes reducing
variance or adding genuine information transfer very differently from changes that just fit the
same information better, a pattern that held for the rest of the project.

## v1 — Single LightGBM — 0.55331
**~2026-08-15 · baseline**

One pooled gradient-boosted tree model across all 1,782 series, `log1p(sales)` as the target
(so training loss is exactly the competition metric), lags and rolling means shifted at least
16 days for legality against the full forecast horizon, holiday names (not a binary flag — a
single flag averages a −93% New Year's effect with a +86% Christmas-bridge effect to a
meaningless +1%) and known-future promotion status as inputs. **23.8% better than the strongest
naive baseline** (0.52063, same-weekday mean of the last 8 weeks) even before any of the above.

---

## What didn't make the cut

Roughly 40 tested ideas never shipped, spanning six categories: feature engineering (region-
scoped promotion, EMA smoothing, trend/acceleration features, conditional promo anchors,
national sales aggregation, rolling medians, run-length features — each reconstructible from
existing splits or too small to clear the noise floor); model diversity (XGBoost, CatBoost,
TiDE, linear/Fourier regression, Ridge/Lasso — each too correlated with the tree model or too
weak on its own); blend-weight refinements beyond v9 (by forecast horizon, by promotion state,
by store age, by sales-volume decile — none held up under leave-one-out testing the way the
three categories in v9 did); training-time changes (Tweedie/Poisson/Huber objectives,
sample-weight decay, oil price and its interactions — the last killed by simple arithmetic: the
2015–16 oil crash never overlaps any window this model can measure or deploy against); and
structural ideas (hierarchical reconciliation, integer rounding, stock-out-run removal — each
ruled out by computing its theoretical ceiling before building it, a technique that took minutes
and saved a notebook each time).

The full reasoning for each is in the private engineering log; the ones with real leaderboard
consequences are in the [README](README.md#results).
