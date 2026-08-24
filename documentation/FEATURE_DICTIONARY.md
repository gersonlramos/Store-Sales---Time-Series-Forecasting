# Feature Dictionary

The 60 inputs my production model receives, what each one means, where it comes from, and —
the column that matters most on a forecasting problem — **why I consider it legal to know at
forecast time**.

Target: `log1p(sales)`, inverted with `expm1` and clipped at zero. Row grain: one
`(date, store_nbr, family)` — 54 stores × 33 families = 1,782 series per day.

---

## The legality rule, which governs half this document

A target date `d` forecast at horizon `h` has **forecast origin `o = d − h`**. I only consider a
feature legal if its value is knowable at `o`.

For a backward-looking feature `lag_k` — the sales figure from `d − k` — that means:

```
d − k ≤ d − h    ⟺    k ≥ h
```

**A feature must be at least as old as the horizon it serves.** Getting this backwards leaks
future sales into training, which produces an excellent holdout score and a collapsed
leaderboard. I assert it in the production notebook rather than trusting the comment.

Two consequences run through the tables below:

- **I parameterise sales-derived features by `min_lag`**, not fixed. The model serving day 1
  uses `lag_1`; the model serving days 5–16 uses `lag_16`. Names below show the `lag ≥ 16`
  (production) variant, with the ladder rule given once.
- **I exempt `onpromotion` entirely.** Promotion schedules are decided by the chain and
  published in `test.csv` for the whole forecast window, so current-day and *forward* promo
  values are known at the origin. This is the single biggest structural advantage I found in
  this dataset, and the promo block below is where my project's largest gain came from.

### Four models, four lag ladders

| Serves | `min_lag` | Lag ladder actually used |
|---|---|---|
| day 1 | 1 | 1, 2, 3, 4, 5, 6, 7, 13, 20, 34, 48 |
| day 2 | 2 | 2, 3, 4, 5, 6, 7, 8, 14, 21, 35, 49 |
| days 3–4 | 4 | 4, 5, 6, 7, 8, 9, 10, 16, 23, 37, 51 |
| days 5–16 | 16 | 16, 17, 18, 19, 20, 21, 22, 28, 35, 49, 63 |

My ladder is always `min_lag + (0,1,2,3,4,5,6,12,19,33,47)` — same shape, shifted. I compute
rolling windows likewise on `sales.shift(min_lag)`, and same-weekday averages start at the first
multiple of 7 at or beyond `min_lag`, so they never mix weekdays.

---

## 1. Series identity — 6 features

Lets a thin series borrow strength from similar ones, and lets my model hold 1,782 different
baseline levels in one set of parameters.

| Feature | Type | Definition | Source |
|---|---|---|---|
| `store_nbr` | int16 | Store number, 1–54 | `train.csv` |
| `family` | **categorical** | Product category, 33 levels | `train.csv` |
| `city` | **categorical** | Store city, 22 levels | `stores.csv` |
| `state` | **categorical** | Store province, 16 levels | `stores.csv` |
| `store_type` | **categorical** | Store format A–E, 5 levels | `stores.csv` |
| `cluster` | int16 | Favorita's own store grouping, 1–17 | `stores.csv` |

All static — no time dimension, so no legality question for me here. I pass these to LightGBM
as **native categoricals**, not one-hot: the library splits on category subsets directly, which
is both faster and more expressive than dummy columns for tree models.

> I tested data-driven clustering of stores by sales behaviour against these and rejected it —
> it explained 89.3% of store-volume variance against `cluster`'s 88.3%, and `store_nbr` already
> carries the same information at full resolution. Any clustering I build is a lossy compression
> of a feature the model already has.

## 2. Recent sales level — 24 features

**~95% of my model's total decision weight.** Everything here derives from `log1p(sales)` on my
earthquake-repaired panel, shifted by `min_lag`.

| Feature | Definition | Window | Legality |
|---|---|---|---|
| `lag_16` … `lag_63` (11) | Sales on a single past day | see ladder above | `k ≥ min_lag ≥ h` ✅ |
| `rmean_7/14/28/56/112` | Mean over the last N days | ends `min_lag` days before `d` | ✅ |
| `rstd_14/28` | Standard deviation over N days — volatility | ends at `min_lag` | ✅ |
| `rmax_28` | Maximum over 28 days — recent peak | ends at `min_lag` | ✅ |
| `dow_mean_4/8` | Mean of the same weekday over the last 4 / 8 occurrences | starts at the first multiple of 7 ≥ `min_lag` | ✅ |
| `zfrac_28/112` | Fraction of days with zero sales | ends at `min_lag` | ✅ |
| `days_since_sale` | Days since this series last sold anything, capped at 999 | as of `min_lag` days before `d` | ✅ |

**I make every rolling window a multiple of 7 by design.** A 7-day window contains exactly one
of each weekday, so its value doesn't depend on where in the week I'm standing. A 3-day window
would be weekday-contaminated — and with Sunday outselling Thursday by 63%, I found that
matters. (I tested `rmean_3`: it helps only in the near-horizon models, where it's genuinely
fresh, and by a margin too small for me to adopt.)

**I found these features heavily redundant with each other, and I'm fine with that.** Dropping
`rmean_7` — 69% of gain-based importance — cost me only +0.0025, because `rmean_28` and `lag_21`
absorb it almost perfectly. Gain-based importance measures split usage, not unique information.

## 3. Promotions — 16 features

`onpromotion` is the count of items on promotion for that store-family-day, given in
`test.csv` for the entire forecast window. I `log1p`-transform every value below.

### Own-series promo

| Feature | Definition | Legality |
|---|---|---|
| `promo` | Today's promo count | ✅ published in `test.csv` |
| `promo_rmean_7/28` | Mean over the last N days | ✅ backward-looking |
| `promo_lag_16` | Promo count 16 days ago | ✅ |
| `promo_rel_112/28` | Today's count **minus its own N-day norm** | ✅ |
| `promo_lead_1/2/3/7` | Promo count 1/2/3/7 days **ahead** | ✅ known-future — ⚠️ see edge note |
| `promo_fwd7` | Mean of the next 7 days | ✅ known-future |

### Aggregated promo — campaigns run above the single-series level

| Feature | Definition | Legality |
|---|---|---|
| `promo_chain_level` | Mean promo across all 1,782 series that day | ✅ |
| `promo_chain_rel` | Chain level minus its own 112-day norm | ✅ |
| `fam_promo_rel` | This family's chain-wide promo minus its 112-day norm | ✅ |
| `fam_promo_fwd7` | This family's next-7-day mean minus its 112-day norm | ✅ known-future |
| `store_promo_rel` | This store's total promo minus its 112-day norm | ✅ |

**Why I use "relative to own norm" rather than raw counts.** Promotional intensity varies by
orders of magnitude between series. A flat count treats a 2→20 jump the same as 200→2000; the
relative form flags both as equally large deviations. Reworking promo into this shape gave me
the largest single gain in the project's history (**−0.0802** on the leaderboard) against a
holdout that predicted it would *hurt*.

**Why only `family` gets a forward version from me.** Campaigns are organised **by product
category** — that's the unit a merchandising decision applies to. I built and tested
store-level and chain-level forward windows too; both lost (+0.0034 and +0.0027), because
averaging unrelated categories' campaigns dilutes the signal. The backward `_rel` features
survive at all three scopes because they measure something else: whether this store or the
chain is in an unusually promotional period at all.

> ### ⚠️ Edge behaviour of the forward features — I audited this, it's documented, not a defect
>
> `promo_lead_k` looks `k` days into the future, so on the **last `k` days of the forecast
> window there's nothing to look at**. I leave those rows **NaN**:
>
> | Feature | NaN rows (of 28,512 test rows) |
> |---|---|
> | `promo_lead_1` | 1,782 — the final day |
> | `promo_lead_2` | 3,564 |
> | `promo_lead_3` | 5,346 |
> | `promo_lead_7` | 12,474 |
>
> **This is the same structural situation that destroyed an earlier feature pair of mine.**
> `promo_max_next16` used `np.nansum`, which returns **0.0** for an all-missing window —
> indistinguishable from "confirmed no promotion coming." My model had never seen that pattern
> in training (every training row sits years from the panel's end, where full lookahead exists)
> and it collapsed the final forecast day to ~30% of correct volume.
>
> The `_lead` features avoid this because **I leave NaN as NaN**: LightGBM routes missing values
> down a learned default branch rather than treating them as a confirmed zero. Honest ignorance,
> not a false statement.
>
> **The residual risk, stated plainly:** my training rows never contain NaN in these columns, so
> the default branch isn't learned from data resembling the test edge. I checked this on the
> shipped forecast — the last days show no volume collapse, day-over-day continuity holds, and
> horizon 16 isn't an outlier when I compare the forecast against independent models. **No
> evidence of harm, but I treat this as a property to re-audit whenever the forward reach
> changes.**

## 4. Calendar — 8 features

Derived from the date alone, so knowable arbitrarily far ahead. All legal by construction.

| Feature | Definition |
|---|---|
| `dow` | Day of week, 0=Monday |
| `day` | Day of month |
| `month`, `year`, `dayofyear` | Calendar position |
| `is_weekend` | Saturday or Sunday |
| `days_to_month_end` | Days remaining in the month |
| `payday_window` | Ecuador's public sector pays on the 15th and the last day of the month; this flags the 1st–3rd, 15th–17th, and the final two days |

Raw `dow` and `day` rank low in importance for me (below 0.1% each) — not because weekday
doesn't matter, but because `dow_mean_4/8` already encode *this series'* historical Tuesday
level in a single number, which is far cheaper for a tree than reconstructing it from
`dow × store_nbr × family` splits.

## 5. Holidays — 6 features

From `holidays_events.csv`, after I clean it: I drop `transferred=True` holidays (they moved),
promote `Transfer` rows to real holidays, keep `Bridge`/`Additional`, separate out `Work Day`,
and keep `Event` as its own flag. I join by locale — national to all stores, local on
`(date, city)`. I **drop** regional holidays: ≤5 observations per series isn't enough for me to
trust.

| Feature | Type | Definition |
|---|---|---|
| `nat_hol_name` | **categorical** | Name of the national holiday, 23 levels incl. `"none"` |
| `loc_hol_name` | **categorical** | Name of the city holiday, 26 levels incl. `"none"` |
| `work_day` | int8 | A working Saturday repaying a bridge day |
| `is_event` | int8 | Non-holiday event (Black Friday, World Cup) |
| `days_since_nat` | int16 | Days since the last national holiday, capped at 30 |
| `days_to_nat` | int16 | Days until the next one, capped at 30 |

**I found the names matter, a binary flag doesn't.** Pooled, "is it a holiday" is worth ×1.01 —
it averages `Primer dia del ano` (**×0.07**, every store shut, the largest calendar effect I
found in the dataset) with `Puente Navidad` (**×1.86**). Encoding the name is worth −0.0047 on
national holiday rows for me. The holiday file runs to 2017-12-26, so my whole forecast window
is covered.

My forecast window contains **exactly one holiday**: `Fundacion de Ambato`, 2017-08-24, local
to Ambato — 66 rows of 28,512. I don't find its effect reliably estimable (4 occurrences, 95% CI
×0.64–×1.22, which contains 1.00), so I don't hard-code a multiplier for it.

---

## Post-processing, applied after the models predict

Not a feature, but part of the contract I keep between model output and submitted forecast.

**I force discontinued lines to exactly zero.** A pooled model shares parameters across all
1,782 series and therefore can't emit exact zero; a combination with no sales in over a year
still comes out of mine with a small positive number. I set any series with **no sales at all
in the preceding 365 days** — 65 combinations, 1,040 rows, 3.6% of the forecast and 0.01% of the
units — to 0.0. Worth **−0.0153** on the leaderboard for me.

I set the threshold deliberately cautiously. It sits mid-plateau (330–547 days all give the same
gain with zero false positives); tightening it to 300 days *reverses the sign*, because wrongly
zeroing a row that sells 3 units costs about seven times what correctly zeroing a dead one saves.

---

## Data sources

| File | Used for | Notes |
|---|---|---|
| `train.csv` | target, all sales-derived features, `onpromotion` history | 3M rows; `onpromotion` is identically zero before 2014-04-01 — a recording artefact, not an absence of promotions |
| `test.csv` | `onpromotion` for the forecast window | what makes my forward promo features legal |
| `stores.csv` | `city`, `state`, `store_type`, `cluster` | static |
| `holidays_events.csv` | the holiday block | needs the cleaning rules above; I can't join it as-is |
| `oil.csv` | **not used** | I rejected it five ways: raw correlation (−0.02 detrended), a trained model, a lagged/regime reformulation, a live leaderboard A/B (+0.00083), and a family-level heterogeneity test |
| `transactions.csv` | **not used** | r = 0.84 at store-day level but ~0.25 at store×family; redundant with store identity plus recent sales. I shipped it once as a leaderboard test: +0.00307, reverted |

## Two panel corrections I apply before computing any feature

**Christmas.** 25 December 2013–2016 is absent from `train.csv` — every store shut. Left as a
gap it silently distorts every rolling window that spans it. I reinsert those dates as
zero-sales days.

**The April 2016 earthquake.** The week of 2016-04-16 carries relief-buying that isn't normal
demand. I repair it **at source** — I replace it with each series' same-weekday median from the
surrounding 8 weeks — rather than flag it with a dummy. My reason: the shock enters through two
doors — as a target, and as an input to every lag and rolling window that reaches back into it,
which at a 112-day window shifted 16 days extends contamination into August 2016. A dummy closes
only the first door. Measured effect on my holdout: none (the dummy variant is in fact the
worst of the three treatments I tested). I keep the repair as correctness insurance, not as
something I present as a source of performance.
