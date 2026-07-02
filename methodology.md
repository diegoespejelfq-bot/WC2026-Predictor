# Football Match Outcome Prediction — Methodology Summary

## Dataset
International football results, 1872–2017+ (Kaggle: `martj42/international-football-results-from-1872-to-2017`), downloaded via `kagglehub` and loaded into **Apache Spark** on a Dataproc cluster. Columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`.

## 1. Environment Setup
- Loaded dataset locally via `kagglehub.dataset_download()`.
- Created a `SparkSession` (not auto-available outside Databricks).
- Resolved a distributed-filesystem issue: `kagglehub` downloads only to the **driver node's local disk**, which worker nodes on a multi-node Dataproc cluster cannot access. GCS was unavailable due to IAM restrictions, so the CSV was instead pushed to **HDFS** (`hdfs dfs -put`) and read from there via `spark.read.csv("hdfs:///...")`.

## 2. Data Cleaning & Filtering
- Filtered out matches prior to **January 1, 2000**, keeping only the modern era (`F.col("date") >= "2000-01-01"`), after casting the `date` column from string to proper `DateType`.
- Manually **imputed missing scores** for several recent 2026 FIFA World Cup matches (e.g. Mexico 2–0 Ecuador, France 3–0 Sweden) using `F.when()` conditions matched on `date` + `home_team` + `away_team`, since these results weren't yet in the source dataset.

## 3. Feature Engineering — Home/Away/Neutral Splits
- Reshaped each match into **two team-perspective rows** (one for the home side, one for the away side), tagging each with a `venue` label (`home`, `away`, or `neutral` based on the `neutral` flag).
- Computed **average goals scored and conceded per team, per venue type**, both as a flat table and pivoted into one row per team with `home_avg_scored`, `away_avg_conceded`, etc.

## 4. Recency Weighting
- Recognized that an all-time average treats old and recent matches equally, which understates current team form.
- Applied **exponential time decay** weighting: `weight = exp(-days_ago / half_life × ln 2)`, where `days_ago` is the gap between a match and a reference date.
- Selected a **730-day (2-year) half-life**, aligned with the natural ~4-year World Cup roster cycle — recent matches count more, but enough historical depth is retained for teams that play infrequently.
- Compared weighted vs. plain averages to sanity-check the impact of the decay parameter before committing to it.

## 5. Leakage-Free Feature Construction
- Identified that using *all-time* team averages (including future matches) as model inputs would leak information — a match's outcome shouldn't be predictable using stats computed partly from itself or from matches after it.
- Built a **self-join on team + prior match date only** (`t2.date < t1.date`), computing each team's *weighted* pre-match average goals scored/conceded strictly from matches **before** the match being predicted.
- Joined these pre-match features back onto each match for both the home and away team, producing a clean training table (`model_data`).
- Filtered out matches where either team had fewer than 5 prior matches, to avoid unstable early-history estimates.

## 6. Modeling — Poisson Regression
- Since goals are count data, used **Poisson regression** (`GeneralizedLinearRegression(family="poisson", link="log")` in Spark ML) rather than standard linear regression — the standard approach in football analytics (Dixon-Coles style models).
- Trained **two separate models**:
  - **Home goals model**: predictors = home team's pre-match scoring average, away team's pre-match conceding average, neutral-venue flag.
  - **Away goals model**: predictors = away team's pre-match scoring average, home team's pre-match conceding average, neutral-venue flag.

## 7. Prediction
- For a given matchup (e.g. Mexico home vs. England away), pulled each team's **latest available pre-match weighted averages** (most recent row per team, ordered by date) rather than arbitrary placeholder values.
- Fed these into both trained models to produce **expected goals (λ)** for each side — e.g. Mexico's expected goals as the home team, England's expected goals as the away team.
- Flagged the importance of setting the `neutral` flag correctly per match, since it materially shifts the home-advantage term.

## Next Steps (Not Yet Implemented)
- Convert the two expected-goals (λ) values into **win/draw/loss probabilities**, typically via a Poisson score-matrix (simulating scorelines from both λs) or a Dixon-Coles low-score correlation adjustment.
- Consider backtesting different `half_life_days` values against held-out recent matches to empirically tune recency weighting rather than choosing it by intuition alone.
- Cast `home_score`/`away_score` cleanly from string to integer dataset-wide, handling `'NA'` placeholders for not-yet-played matches.
