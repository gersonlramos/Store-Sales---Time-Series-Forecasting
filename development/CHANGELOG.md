# Version History

How I took the model from **0.55331** to **0.38930** on the public leaderboard — nine changes I
shipped, in the order I made them, each with the reasoning that justified it to me at the time.
I note rejected attempts inline where they explain why my path bends the way it does; the full
account of every rejection I made (roughly 40, each with a measured reason) lives in the
[README's measurement record](../README.md#what-makes-this-project-unusual-the-measurement-record)
and my private engineering log.

Every score below is the **public leaderboard** — this competition has no private leaderboard,
so the public score is the only real-world number I have. "Local" is the held-out fortnight
inside `train.csv`, which I use to decide what to submit before spending a submission on it.

---

## v9 — Family-gated blend weight — **0.38930** (current)
**2026-08-22 · −0.00149 from v8**

Three product categories — `SCHOOL AND OFFICE SUPPLIES`, `HOME APPLIANCES`, `AUTOMOTIVE` — use a
different direct/recursive split than the uniform 70/30 every other category gets (35/65,
90/10, and 50/50 respectively). I found this by optimising the split independently on three
separate historical periods and keeping only categories where the same split helped in **all
three**, never picked by looking at the window it would be scored on. My local estimate was
−0.00191, blind to the real leaderboard window. It transferred at 0.78× — smaller than v8's own
transfer ratio, but the same sign, and the third result in a row from this validation discipline
(leave-one-out across independent windows) with zero reversals for me.

*I rejected two things the same week, tested with the same rigor:* a mixed L2+Huber training
objective (I measured −0.00197 locally, 5/5 seeds; **it reversed to +0.00241 on the real
leaderboard**) and `rmean_3` — a third rolling-mean window for the near-horizon models (I
measured −0.00244, 5.5σ; it landed flat, +0.00015, on the real window). I'd validated both on a
*single* window, unlike the family-gated weight's three — the likely reason one reversed, one
went flat, and this one held for me.

## v8 — Recursive per-family blend — 0.39079
**2026-08-20/21 · −0.00507 from v7**

The first "second model" — of seven candidates I tried across three weeks — that was both
genuinely different from my tree-based direct model (residual correlation 0.917, versus
0.97–0.99 for every other tree-based variant I tried) *and* good enough on its own to earn real
blend weight (RMSLE 0.399 solo, versus a 0.10+ gap for every neural or linear alternative I
tried). I trained one small model per product category, predicting one day at a time and feeding
each prediction back in — the opposite failure mode from my direct model: fresh but increasingly
speculative, where direct is reliable but stale. I blended the two 70/30 in log space.

## v7 — Dormant-series hard zero — 0.39586
**2026-08-19 · −0.01527 from v6, the largest single transfer in my project (5.6× the local
measurement)**

Not a new feature — a post-processing rule I added. A pooled model sharing parameters across
1,782 series structurally can't output an exact zero, so any store/category combination with no
sales in the trailing year still came out of mine with a small positive forecast. I found this
by comparing my submission against a leading public one row by row: one difference — 2,032 rows
where the public blend predicted exactly zero and mine didn't — explained most of my remaining
gap to it.

## v6 — Per-horizon models — 0.41113
**2026-08-19 · −0.00961 from v5, transferred at 4.4×**

Four models instead of one, each using the freshest sales data its forecast day can legally
see. My single model predicting all 16 days had to obey the horizon-16 staleness rule even on
day 1, when far fresher data was available and legal for it to use. No new columns — the same
features, just without a staleness constraint the near-term forecasts never needed.

## v5 — Round cap 1200→2400, patience 50→150 — 0.42074
**~2026-08-18 · −0.00019 from v4**

My model had been stopping before convergence: at the old patience setting it halted around
900 trees; it now stops on its own near 1,725. A clean, paired, 3/3-seed local win for me of
−0.00263 — but it transferred at only ~7% of that to the real leaderboard, the first sign I saw
that hyperparameter-tuning changes behave very differently from changes that give the model new
information.

## v4 — Family/store promotion intensity + tuning — 0.42093
**~2026-08-17/18 · −0.04171 from v3**

I extended the relative-promotion idea from v3 to the family and store scope — not just "is this
series promoting more than usual" but "is this whole product category, or this whole store,
promoting more than usual right now" — and lowered `min_data_in_leaf` from 200 to 50 (a real,
monotonic local win for me, independent of the promotion features).

## v3 — Promotion rework: relative intensity + known-future leads — 0.46264
**~2026-08-17 · −0.0802 from v2, the largest single leaderboard gain in my project**

The single biggest result I've gotten here, and the one hardest to trust in advance. I'd used
`onpromotion` as a raw count; I rebuilt it as *intensity relative to each series' own 112-day
norm*, plus 1/2/3/7-day-ahead leads (legal because `test.csv` gives promotion status for the
entire forecast window, unlike sales). Locally this looked to me like a **wash or a loss** — my
2015/2016 backtests predicted +0.0016 to +0.0029 (worse). 2017's actual promotional ramp into
the test window turned out to be the largest on record, a pattern no backtest year I had
contained at comparable scale. The gap between my local and leaderboard scores closed by roughly
80% across this change and the two that followed it — the clearest evidence I've found that my
local validation reliably catches regressions but can badly underestimate a genuine win tied to
information no backtest year I have happens to contain.

*I'd built a closely related feature — a promo signal reaching 16 days into the forecast
horizon — backtested it, and it looked like a clean win before this one shipped. I never
submitted it*: a visual check of the forecast chart (not the summary statistics) caught a
last-day volume collapse to ~30% of correct, caused by `np.nansum` over an all-missing slice at
the panel's edge silently returning 0.0 instead of "unknown." No backtest year I had could have
shown me this — every backtest sits years from the panel's true edge. I now check the chart
before every model change of this kind, not just this one.

## v2 — 3-seed ensemble — 0.54284
**~2026-08-16 · −0.01047 from v1, transferred at ~3.5× its local measurement**

I averaged three LightGBM fits differing only by random seed. A small, well-understood variance
reduction — notable mainly because the transfer ratio here first hinted to me that changes
reducing variance or adding genuine information transfer very differently from changes that just
fit the same information better, a pattern that held for the rest of my project.

## v1 — Single LightGBM — 0.55331
**~2026-08-15 · baseline**

One pooled gradient-boosted tree model across all 1,782 series, `log1p(sales)` as the target
(so my training loss is exactly the competition metric), lags and rolling means shifted at least
16 days for legality against the full forecast horizon, holiday names (not a binary flag — I
found a single flag averages a −93% New Year's effect with a +86% Christmas-bridge effect to a
meaningless +1%) and known-future promotion status as inputs. **23.8% better than the strongest
naive baseline I could find** (0.52063, same-weekday mean of the last 8 weeks) even before any
of the above.

---

## What didn't make the cut

I tested roughly 40 ideas that never shipped, spanning six categories: feature engineering
(region-scoped promotion, EMA smoothing, trend/acceleration features, conditional promo anchors,
national sales aggregation, rolling medians, run-length features — each reconstructible from
existing splits or too small to clear my noise floor); model diversity (XGBoost, CatBoost, TiDE,
linear/Fourier regression, Ridge/Lasso — each too correlated with my tree model or too weak on
its own); blend-weight refinements beyond v9 (by forecast horizon, by promotion state, by store
age, by sales-volume decile — none held up under the leave-one-out testing I gave the three
categories in v9); training-time changes (Tweedie/Poisson/Huber objectives, sample-weight decay,
oil price and its interactions — I killed the last one by simple arithmetic: the 2015–16 oil
crash never overlaps any window I can measure or deploy against); and structural ideas
(hierarchical reconciliation, integer rounding, stock-out-run removal — each I ruled out by
computing its theoretical ceiling before building it, a technique that took me minutes and saved
a notebook each time).

My full reasoning for each is in my private engineering log; the ones with real leaderboard
consequences are in the [README](../README.md#results).

---

## Closing round — four more ideas, post-v9, none adopted
**2026-08-22**

After v9 shipped, four more ideas were raised — sharp enough that I investigated each one
properly rather than dismissing it by analogy to something I'd already closed. None survived
paired testing:

- **Completing the per-horizon design** (a model for every day 5–16, not one model for the
  whole range). It looked real at first — a single-day exploratory read showed a 3/3-seed, 2.7σ
  win at day 8. It didn't survive contact with a larger, more realistic sample: scored as the
  actual bucket redesign it would be (days 5–8 together, 7,128 rows instead of 1,782), the
  effect fell to 1.0σ with no consistent pattern across the four days. The lesson I'm keeping:
  a result measured on an unusually small slice needs to be re-measured on the realistic
  deployment-sized slice before I trust it — the noise floor moves with sample size, it isn't
  a fixed number.
- **Auditing the forward-promo features at the panel's true edge**, where they go NaN on ~44%
  of real test rows in a pattern my training data never exhibits. My result was ambiguous rather
  than reassuring or alarming (one of three windows even improved under the audit condition),
  and unlike the one confirmed edge-of-panel bug I've found in this project, nothing here showed
  the sharp, unambiguous signature a real problem of that kind would produce.
- **Family promo coverage** — how many stores in a category are running any promotion, distinct
  from total promo intensity. The pattern motivating this was real and I verified it directly
  against the competition data (some categories show 2–5× more stores promoting in the forecast
  window than their own recent norm), but neither a level nor a relative version of the feature
  I built cleared even the weakest bar I use to justify further testing.
- **Promotion streak length** — I didn't run this one. It's a strictly more complex function of
  the same raw near-term promotion signal I'd already tested as essentially unused by the model.

## Project status: complete

**0.38930 is my final result.** I tested every well-motivated idea raised after it — across two
full rounds of investigation — with the same discipline that got v9 shipped, and none held up.
That includes ideas with real, independently-verified patterns behind them; I learned that
verifying a pattern exists in the data and finding that my model can profitably use it turned
out to be two different questions more often than not. I don't have any further changes planned.
