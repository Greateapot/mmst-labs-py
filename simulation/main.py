"""Monte Carlo model of a short-stay rental business.

The script expects four REDATA CSV files to exist in --data-dir:
    months.csv, calendar.csv, weekdays.csv, stay_length.csv

No data is downloaded. All generated files are written under --output-dir
(--output-dit is accepted as an alias for compatibility with the task wording).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution


# ---------------------------------------------------------------------------
# Configuration and input preparation
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
  snapshot_date: str = "2026-11-02"
  start_date: str | None = "2026-11-02"
  horizon_days: int = 90
  units: int = 10

  # These two fields are overwritten from the latest complete month in
  # months.csv. Their values in an older config file are not authoritative.
  source_adr_rub: float = 0.0
  source_occupancy_pct: float = 0.0

  owner_block_share: float = 0.10
  price_elasticity: float = 1.20
  demand_shock_sigma: float = 0.25

  cleaning_mean_hours: float = 2.3
  cleaning_lognormal_sigma: float = 0.35
  cleaning_shift_hours: float = 8.0
  cleaning_shift_start_hour: float = 9.0
  cleaning_shift_end_hour: float = 17.0

  rent_per_unit_month_rub: float = 35000.0
  cleaner_cost_per_day_rub: float = 2200.0
  platform_fee_pct: float = 0.15
  tax_pct: float = 0.06
  variable_cost_per_occupied_night_rub: float = 450.0
  cleaning_supply_per_turnover_rub: float = 200.0
  opening_cash_rub: float = 500000.0

  booking_request_window_start_hour: float = 12.0
  booking_request_window_end_hour: float = 20.0

  # The REDATA tail bucket is open-ended. This is an explicit modeling cap.
  stay_tail_cap_nights: int = 30

  monte_carlo_runs: int = 300
  optimization_runs: int = 8
  sensitivity_runs: int = 50
  dynamic_runs: int = 60
  optimizer_maxiter: int = 4
  optimizer_popsize: int = 5
  risk_aversion: float = 0.25
  random_seed: int = 20261109

  @classmethod
  def from_json(cls, path: Path) -> ModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
      raise TypeError("The model config must be a JSON object")
    allowed = set(cls.__dataclass_fields__)
    # Old config files may contain obsolete source ADR/occupancy values;
    # those fields are retained for compatibility and then recalibrated.
    unknown = sorted(set(raw) - allowed)
    if unknown:
      raise ValueError(f"Unknown model config fields: {', '.join(unknown)}")
    return cls(**raw)

  def validate(self) -> None:
    if self.horizon_days < 1:
      raise ValueError("horizon_days must be >= 1")
    if self.units < 1:
      raise ValueError("units must be >= 1")
    if not 0.0 <= self.owner_block_share < 1.0:
      raise ValueError("owner_block_share must be in [0, 1)")
    if self.demand_shock_sigma < 0:
      raise ValueError("demand_shock_sigma must be >= 0")
    if self.cleaning_mean_hours <= 0:
      raise ValueError("cleaning_mean_hours must be > 0")
    if self.cleaning_lognormal_sigma < 0:
      raise ValueError("cleaning_lognormal_sigma must be >= 0")
    if not (0 <= self.cleaning_shift_start_hour < self.cleaning_shift_end_hour <= 24):
      raise ValueError("Invalid cleaning shift bounds")
    if not math.isclose(
      self.cleaning_shift_end_hour - self.cleaning_shift_start_hour,
      self.cleaning_shift_hours,
      abs_tol=1e-8,
    ):
      raise ValueError(
        "cleaning_shift_hours must equal shift_end_hour - shift_start_hour"
      )
    if not (
      0
      <= self.booking_request_window_start_hour
      < self.booking_request_window_end_hour
      <= 24
    ):
      raise ValueError("Invalid booking request time window")
    if self.stay_tail_cap_nights < 15:
      raise ValueError("stay_tail_cap_nights must be >= 15")
    if (
      min(
        self.monte_carlo_runs,
        self.optimization_runs,
        self.sensitivity_runs,
        self.dynamic_runs,
      )
      < 1
    ):
      raise ValueError("All replication counts must be >= 1")


@dataclass
class Decision:
  price_multiplier: float = 1.0
  weekend_markup: float = 0.0
  cleaners: int = 1

  def __post_init__(self) -> None:
    if not 0.8 <= self.price_multiplier <= 1.25:
      raise ValueError("price_multiplier must be in [0.8, 1.25]")
    if not 0.0 <= self.weekend_markup <= 0.30:
      raise ValueError("weekend_markup must be in [0, 0.30]")
    if int(self.cleaners) != self.cleaners or not 1 <= self.cleaners <= 3:
      raise ValueError("cleaners must be an integer in [1, 3]")


@dataclass
class MarketData:
  months: pd.DataFrame
  calendar: pd.DataFrame
  weekdays: pd.DataFrame
  stays: pd.DataFrame
  monthly_summary: pd.DataFrame
  city_id: str
  stay_mean: float
  stay_decay: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Monte Carlo simulation for a short-stay rental portfolio"
  )
  parser.add_argument(
    "--config",
    required=True,
    type=Path,
    help="Path to JSON model configuration (required)",
  )
  parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data"),
    help="Directory with months.csv, calendar.csv, weekdays.csv and stay_length.csv (default: data)",
  )
  parser.add_argument(
    "--output-dir",
    dest="output_dir",
    type=Path,
    default=Path("out"),
    help="Output directory; results/ and figures/ are created inside it (default: out)",
  )
  return parser.parse_args()


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
  output_dir = output_dir.expanduser().resolve()
  results_dir = output_dir / "results"
  figures_dir = output_dir / "figures"
  results_dir.mkdir(parents=True, exist_ok=True)
  figures_dir.mkdir(parents=True, exist_ok=True)
  return results_dir, figures_dir


def _read_csv(
  data_dir: Path, filename: str, required_columns: set[str]
) -> pd.DataFrame:
  path = data_dir / filename
  if not path.is_file():
    raise FileNotFoundError(f"Required source file not found: {path}")
  frame = pd.read_csv(path, encoding="utf-8-sig")
  missing = required_columns - set(frame.columns)
  if missing:
    raise ValueError(f"{filename}: missing columns: {', '.join(sorted(missing))}")
  if frame.empty:
    raise ValueError(f"{filename}: file contains no rows")
  return frame


def _numeric_columns(
  frame: pd.DataFrame, filename: str, columns: list[str], allow_missing: bool = False
) -> pd.DataFrame:
  result = frame.copy()
  for column in columns:
    result[column] = pd.to_numeric(result[column], errors="coerce")
    if not allow_missing and result[column].isna().any():
      raise ValueError(
        f"{filename}: column {column!r} contains missing/non-numeric values"
      )
  return result


def _normalize_bucket(value: Any) -> str:
  text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
  text = text.replace("—", "-").replace("–", "-").replace("−", "-")
  text = re.sub(r"\s+", " ", text)
  return text


def _bucket_values(label: Any, tail_cap: int) -> np.ndarray:
  key = _normalize_bucket(label)
  # These are the actual labels in the supplied REDATA export.
  mapping: dict[str, np.ndarray] = {
    "1 ночь": np.array([1], dtype=int),
    "2 ночи": np.array([2], dtype=int),
    "3 ночи": np.array([3], dtype=int),
    "4-7 ночей": np.arange(4, 8, dtype=int),
    "8-14 ночей": np.arange(8, 15, dtype=int),
    "больше двух недель": np.arange(15, tail_cap + 1, dtype=int),
  }
  if key not in mapping:
    raise ValueError(f"Unknown stay-length bucket: {label!r}")
  return mapping[key]


def build_stay_distribution(
  stays: pd.DataFrame, tail_cap: int
) -> tuple[pd.DataFrame, float, float]:
  """Expand REDATA duration buckets to integer nights.

  For the interval buckets, values are sampled from an exponentially
  decreasing discrete distribution. Its single decay parameter is calibrated
  so that the weighted sample mean matches REDATA's average_nights field.
  The open-ended tail is capped at tail_cap nights (configurable assumption).
  """
  frame = stays.copy().reset_index(drop=True)
  frame = _numeric_columns(frame, "stay_length.csv", ["share_pct", "average_nights"])
  if (frame["share_pct"] < 0).any() or frame["share_pct"].sum() <= 0:
    raise ValueError(
      "stay_length.csv: share_pct must be non-negative and have a positive sum"
    )
  if frame["bucket"].duplicated().any():
    raise ValueError("stay_length.csv: bucket labels must be unique")

  frame["model_values"] = [_bucket_values(label, tail_cap) for label in frame["bucket"]]
  shares = frame["share_pct"].to_numpy(dtype=float)
  shares = shares / shares.sum()
  target_mean = float(np.average(frame["average_nights"], weights=shares))

  supports = frame["model_values"].tolist()

  def probabilities_for(values: np.ndarray, beta: float) -> np.ndarray:
    log_weights = -beta * (values - values[0]).astype(float)
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    return weights / weights.sum()

  def implied_mean(beta: float) -> float:
    conditional_means = [
      float(np.dot(values, probabilities_for(values, beta))) for values in supports
    ]
    return float(np.dot(shares, conditional_means))

  minimum_mean = implied_mean(100.0)
  maximum_mean = implied_mean(-100.0)
  if not minimum_mean - 1e-8 <= target_mean <= maximum_mean + 1e-8:
    raise ValueError(
      "average_nights cannot be reconciled with stay_length buckets and "
      f"stay_tail_cap_nights={tail_cap}: target={target_mean:.3f}, "
      f"feasible=[{minimum_mean:.3f}, {maximum_mean:.3f}]"
    )
  beta = float(brentq(lambda value: implied_mean(value) - target_mean, -100.0, 100.0))
  frame["model_probabilities"] = [
    probabilities_for(values, beta) for values in supports
  ]
  calibrated_mean = implied_mean(beta)
  return frame, calibrated_mean, beta


def read_market_data(data_dir: Path, config: ModelConfig) -> MarketData:
  """Read the four existing REDATA CSV files; this function performs no download."""
  months = _read_csv(
    data_dir,
    "months.csv",
    {"city_id", "month", "adr_rub", "occupancy_pct", "listings", "snapshot_date"},
  )
  calendar = _read_csv(
    data_dir,
    "calendar.csv",
    {"city_id", "date", "weekday", "price_rub", "occupancy_pct", "snapshot_date"},
  )
  weekdays = _read_csv(
    data_dir,
    "weekdays.csv",
    {"city_id", "weekday", "weekday_num", "occupancy_pct", "price_rub", "weekend"},
  )
  stays = _read_csv(
    data_dir, "stay_length.csv", {"city_id", "bucket", "share_pct", "average_nights"}
  )

  city_sets = [
    set(frame["city_id"].dropna().astype(str).unique())
    for frame in (months, calendar, weekdays, stays)
  ]
  if any(len(values) != 1 for values in city_sets):
    raise ValueError("Each CSV must contain exactly one non-empty city_id")
  city_ids = {next(iter(values)) for values in city_sets}
  if len(city_ids) != 1:
    raise ValueError(f"CSV files refer to different cities: {sorted(city_ids)}")
  city_id = next(iter(city_ids))

  months = _numeric_columns(
    months, "months.csv", ["adr_rub", "occupancy_pct", "listings"], allow_missing=True
  )
  months["month"] = months["month"].astype(str)
  months = months.sort_values(
    ["month", "district"] if "district" in months.columns else ["month"]
  )

  calendar = _numeric_columns(calendar, "calendar.csv", ["price_rub", "occupancy_pct"])
  calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.date
  calendar = calendar.sort_values("date").reset_index(drop=True)
  if calendar["date"].duplicated().any():
    raise ValueError("calendar.csv: duplicate dates found")
  if (
    (calendar["price_rub"] <= 0)
    | (calendar["occupancy_pct"] < 0)
    | (calendar["occupancy_pct"] > 100)
  ).any():
    raise ValueError(
      "calendar.csv: price_rub must be positive and occupancy_pct in [0, 100]"
    )

  weekdays = _numeric_columns(
    weekdays, "weekdays.csv", ["weekday_num", "occupancy_pct", "price_rub", "weekend"]
  )
  weekdays["weekday_num"] = weekdays["weekday_num"].astype(int)
  weekdays = weekdays.sort_values("weekday_num").reset_index(drop=True)
  if weekdays["weekday_num"].tolist() != list(range(7)):
    raise ValueError("weekdays.csv: expected exactly one row for weekday_num 0..6")
  if weekdays["weekday"].duplicated().any():
    raise ValueError("weekdays.csv: weekday labels must be unique")
  if (
    (weekdays["occupancy_pct"] < 0)
    | (weekdays["occupancy_pct"] > 100)
    | (weekdays["price_rub"] <= 0)
  ).any():
    raise ValueError("weekdays.csv: invalid occupancy_pct or price_rub")
  weekdays["weekend"] = weekdays["weekend"].astype(int)

  stays = _numeric_columns(stays, "stay_length.csv", ["share_pct", "average_nights"])
  if not np.isclose(stays["share_pct"].sum(), 100.0, atol=1.0):
    raise ValueError(
      f"stay_length.csv: share_pct sums to {stays['share_pct'].sum():.2f}, expected approximately 100"
    )
  stays, stay_mean, stay_decay = build_stay_distribution(
    stays, config.stay_tail_cap_nights
  )

  monthly_summary = (
    months
    .groupby("month", as_index=False)
    .agg(
      adr_rub=("adr_rub", "mean"),
      occupancy_pct=("occupancy_pct", "mean"),
      adr_districts=("adr_rub", "count"),
      occupancy_districts=("occupancy_pct", "count"),
      listings=("listings", "max"),
    )
    .sort_values("month")
    .reset_index(drop=True)
  )
  complete_months = monthly_summary.dropna(subset=["adr_rub", "occupancy_pct"])
  if complete_months.empty:
    raise ValueError("months.csv has no month with both ADR and occupancy data")

  # Calibrate market anchor from the latest month, averaging district-level
  # observations because REDATA does not provide district weights in this CSV.
  latest = complete_months.iloc[-1]
  config.source_adr_rub = float(latest["adr_rub"])
  config.source_occupancy_pct = float(latest["occupancy_pct"])

  if config.start_date is None:
    requested_start = calendar["date"].iloc[0]
  else:
    requested_start = date.fromisoformat(config.start_date)
  matching = calendar.index[calendar["date"] == requested_start].tolist()
  if not matching:
    raise ValueError(
      f"start_date={requested_start.isoformat()} is absent from calendar.csv; "
      f"available range is {calendar['date'].min()}..{calendar['date'].max()}"
    )
  start_idx = matching[0]
  calendar = (
    calendar
    .iloc[start_idx : start_idx + config.horizon_days]
    .copy()
    .reset_index(drop=True)
  )
  if len(calendar) != config.horizon_days:
    raise ValueError(
      f"calendar.csv has only {len(calendar)} dates from {requested_start}; "
      f"horizon_days={config.horizon_days} requested"
    )
  date_diffs = calendar["date"].diff().dropna()
  if not date_diffs.eq(timedelta(days=1)).all():
    raise ValueError(
      "calendar.csv must contain consecutive daily dates for the selected horizon"
    )
  config.start_date = calendar["date"].iloc[0].isoformat()

  return MarketData(
    months,
    calendar,
    weekdays,
    stays,
    monthly_summary,
    city_id,
    stay_mean,
    stay_decay,
  )


# ---------------------------------------------------------------------------
# Stay sampling and simulation engine
# ---------------------------------------------------------------------------


def _stay_sampler(stays: pd.DataFrame, rng: np.random.Generator) -> int:
  shares = stays["share_pct"].to_numpy(dtype=float)
  row_index = int(rng.choice(len(stays), p=shares / shares.sum()))
  row = stays.iloc[row_index]
  values = np.asarray(row["model_values"], dtype=int)
  probabilities = np.asarray(row["model_probabilities"], dtype=float)
  return int(rng.choice(values, p=probabilities))


def _move_to_shift(t: float, shift_start: float, shift_end: float) -> float:
  day = math.floor(t / 24.0)
  hour = t - day * 24.0
  if hour < shift_start:
    return day * 24.0 + shift_start
  if hour >= shift_end - 1e-10:
    return (day + 1) * 24.0 + shift_start
  return t


def _add_work_hours(
  start: float, work_hours: float, shift_start: float, shift_end: float
) -> float:
  t = _move_to_shift(start, shift_start, shift_end)
  remaining = float(work_hours)
  while remaining > 1e-10:
    day = math.floor(t / 24.0)
    day_end = day * 24.0 + shift_end
    available = max(0.0, day_end - t)
    if remaining <= available:
      t += remaining
      remaining = 0.0
    else:
      remaining -= available
      t = (day + 1) * 24.0 + shift_start
  return t


def simulate_once(
  config: ModelConfig,
  decision: Decision,
  weekdays: pd.DataFrame,
  stays: pd.DataFrame,
  calendar: pd.DataFrame,
  seed: int,
  *,
  demand_scale: float = 1.0,
  cost_scale: float = 1.0,
  cleaning_time_scale: float = 1.0,
  owner_block_share: float | None = None,
  price_elasticity: float | None = None,
) -> dict[str, Any]:
  """Run one replication with request, checkout and cleaning events.

  calendar.price_rub is the day-specific market price forecast. The calendar
  occupancy_pct is retained in input diagnostics, not interpreted as final
  demand: it is the booking/closure pace observed at the snapshot date.
  """
  rng = np.random.default_rng(seed)
  h = config.horizon_days
  n = config.units
  start_date = date.fromisoformat(config.start_date or str(calendar["date"].iloc[0]))
  block_share = (
    config.owner_block_share if owner_block_share is None else owner_block_share
  )
  elasticity = config.price_elasticity if price_elasticity is None else price_elasticity
  if not 0.0 <= block_share < 1.0:
    raise ValueError("owner_block_share must be in [0, 1)")

  weekday_occ = weekdays["occupancy_pct"].to_numpy(dtype=float) / 100.0
  weekend_by_weekday = weekdays["weekend"].to_numpy(dtype=int).astype(bool)
  weekday_mean_occ = float(weekday_occ.mean())
  day_wd = np.array([(start_date + timedelta(days=d)).weekday() for d in range(h)])

  market_prices = calendar["price_rub"].to_numpy(dtype=float)
  market_occupancy_pace = calendar["occupancy_pct"].to_numpy(dtype=float)
  weekend_multiplier = np.where(
    weekend_by_weekday[day_wd], 1.0 + decision.weekend_markup, 1.0
  )
  price_by_day = market_prices * decision.price_multiplier * weekend_multiplier

  mean_stay = float(
    np.average(
      [
        np.dot(
          np.asarray(row.model_values, dtype=float),
          np.asarray(row.model_probabilities, dtype=float),
        )
        for row in stays.itertuples(index=False)
      ],
      weights=stays["share_pct"].to_numpy(dtype=float),
    )
  )
  occupancy_target = config.source_occupancy_pct / 100.0 * (1.0 - block_share)
  lambda_base = n * occupancy_target / mean_stay

  # 0=clean/available, 1=occupied, 2=dirty/cleaning.
  status = np.zeros(n, dtype=np.int8)
  worker_free = np.zeros(decision.cleaners, dtype=float)
  shift_start = config.cleaning_shift_start_hour
  shift_end = config.cleaning_shift_end_hour
  events: list[tuple[float, int, int, int]] = []
  serial = 0

  request_count = accepted = lost = occupied_nights = 0
  revenue_by_day = np.zeros(h, dtype=float)
  occupied_by_day = np.zeros(h, dtype=int)
  turnovers_by_day = np.zeros(h, dtype=int)
  cleaning_turnovers = 0
  cleaning_work_hours = 0.0
  cleaning_waits: list[float] = []

  # The REDATA bucket shares are the arrival-level stay-duration distribution.
  for day_index in range(h):
    weekday_index = day_wd[day_index]
    decision_price_factor = price_by_day[day_index] / market_prices[day_index]
    rate = lambda_base * (weekday_occ[weekday_index] / weekday_mean_occ)
    rate *= decision_price_factor ** (-elasticity) * demand_scale
    sigma = config.demand_shock_sigma
    shock = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma) if sigma > 0 else 1.0
    count = int(rng.poisson(max(0.0, rate * shock)))
    request_count += count
    if count:
      arrival_hours = rng.uniform(
        config.booking_request_window_start_hour,
        config.booking_request_window_end_hour,
        size=count,
      )
      for hour in arrival_hours:
        stay = _stay_sampler(stays, rng)
        serial += 1
        heapq.heappush(events, (day_index * 24.0 + float(hour), 2, serial, stay))

  while events:
    event_time, event_type, _event_id, payload = heapq.heappop(events)
    if event_time >= h * 24.0:
      continue

    if event_type == 0:  # cleaning completed
      status[payload] = 0
      continue

    if event_type == 1:  # checkout; room becomes dirty
      unit = payload
      status[unit] = 2
      cleaning_turnovers += 1
      checkout_day = min(h - 1, max(0, int(event_time // 24.0)))
      turnovers_by_day[checkout_day] += 1
      duration = rng.lognormal(
        mean=math.log(config.cleaning_mean_hours * cleaning_time_scale)
        - 0.5 * config.cleaning_lognormal_sigma**2,
        sigma=config.cleaning_lognormal_sigma,
      )
      worker_index = int(np.argmin(worker_free))
      cleaning_start = _move_to_shift(
        max(event_time, worker_free[worker_index]), shift_start, shift_end
      )
      cleaning_waits.append(max(0.0, cleaning_start - event_time))
      cleaning_finish = _add_work_hours(
        cleaning_start, duration, shift_start, shift_end
      )
      worker_free[worker_index] = cleaning_finish
      cleaning_work_hours += duration
      serial += 1
      heapq.heappush(events, (cleaning_finish, 0, serial, unit))
      continue

    # Incoming same-day booking request.
    available = np.flatnonzero(status == 0)
    if len(available) == 0:
      lost += 1
      continue
    unit = int(rng.choice(available))
    status[unit] = 1
    accepted += 1
    arrival_day = int(event_time // 24.0)
    stay = int(payload)
    checkout_time = (arrival_day + stay) * 24.0 + 11.0
    for night in range(arrival_day, min(arrival_day + stay, h)):
      occupied_nights += 1
      occupied_by_day[night] += 1
      revenue_by_day[night] += price_by_day[night]
    serial += 1
    heapq.heappush(events, (checkout_time, 1, serial, unit))

  gross_revenue = float(revenue_by_day.sum())
  platform_fees = gross_revenue * config.platform_fee_pct * cost_scale
  taxes = gross_revenue * config.tax_pct * cost_scale
  variable_costs = (
    occupied_nights * config.variable_cost_per_occupied_night_rub * cost_scale
  )
  rent_costs = n * config.rent_per_unit_month_rub * (h / (365.25 / 12.0)) * cost_scale
  labour_costs = decision.cleaners * config.cleaner_cost_per_day_rub * h * cost_scale
  cleaning_supplies = (
    cleaning_turnovers * config.cleaning_supply_per_turnover_rub * cost_scale
  )
  total_costs = (
    platform_fees
    + taxes
    + variable_costs
    + rent_costs
    + labour_costs
    + cleaning_supplies
  )
  profit = gross_revenue - total_costs

  daily_platform = revenue_by_day * config.platform_fee_pct * cost_scale
  daily_tax = revenue_by_day * config.tax_pct * cost_scale
  daily_variable = (
    occupied_by_day * config.variable_cost_per_occupied_night_rub * cost_scale
  )
  daily_rent = np.full(h, rent_costs / h)
  daily_labour = np.full(h, labour_costs / h)
  daily_supplies = (
    turnovers_by_day * config.cleaning_supply_per_turnover_rub * cost_scale
  )
  daily_net_cash = (
    revenue_by_day
    - daily_platform
    - daily_tax
    - daily_variable
    - daily_rent
    - daily_labour
    - daily_supplies
  )
  cash_path = config.opening_cash_rub + np.cumsum(daily_net_cash)
  cash_end = float(cash_path[-1]) if len(cash_path) else config.opening_cash_rub
  min_cash_balance = (
    float(cash_path.min()) if len(cash_path) else config.opening_cash_rub
  )

  occupied_rate = occupied_nights / (n * h)
  accepted_rate = accepted / request_count if request_count else 1.0
  cleaning_utilization = cleaning_work_hours / (
    decision.cleaners * h * config.cleaning_shift_hours
  )
  cleaning_utilization = min(1.0, cleaning_utilization)

  return {
    "seed": int(seed),
    "profit_rub": profit,
    "revenue_rub": gross_revenue,
    "total_costs_rub": total_costs,
    "platform_fees_rub": platform_fees,
    "taxes_rub": taxes,
    "variable_costs_rub": variable_costs,
    "rent_costs_rub": rent_costs,
    "labour_costs_rub": labour_costs,
    "cleaning_supplies_rub": cleaning_supplies,
    "occupancy_pct": 100.0 * occupied_rate,
    "requests": request_count,
    "accepted_bookings": accepted,
    "lost_requests": lost,
    "acceptance_pct": 100.0 * accepted_rate,
    "cleaning_turnovers": cleaning_turnovers,
    "cleaner_utilization_pct": 100.0 * cleaning_utilization,
    "mean_cleaning_wait_hours": float(np.mean(cleaning_waits))
    if cleaning_waits
    else 0.0,
    "p95_cleaning_wait_hours": float(np.quantile(cleaning_waits, 0.95))
    if cleaning_waits
    else 0.0,
    "max_cleaning_wait_hours": float(max(cleaning_waits)) if cleaning_waits else 0.0,
    "dirty_or_cleaning_units_at_end": int(np.sum(status == 2)),
    "cash_end_rub": cash_end,
    "minimum_cash_balance_rub": min_cash_balance,
    "cash_insolvency": int(min_cash_balance < 0),
    "daily_net_cash": daily_net_cash,
    "daily_cash_balance": cash_path,
    "daily_revenue": revenue_by_day,
    "daily_occupancy": occupied_by_day,
    "calendar_booking_pace_mean_pct": float(np.mean(market_occupancy_pace)),
    "decision_price_multiplier": decision.price_multiplier,
    "decision_weekend_markup": decision.weekend_markup,
    "decision_cleaners": decision.cleaners,
  }


def run_replications(
  config: ModelConfig,
  decision: Decision,
  weekdays: pd.DataFrame,
  stays: pd.DataFrame,
  calendar: pd.DataFrame,
  seeds: list[int],
  **kwargs: Any,
) -> pd.DataFrame:
  rows = []
  for seed in seeds:
    result = simulate_once(config, decision, weekdays, stays, calendar, seed, **kwargs)
    rows.append({
      key: value for key, value in result.items() if not isinstance(value, np.ndarray)
    })
  return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, scenario: str) -> dict[str, Any]:
  profits = frame["profit_rub"].to_numpy(dtype=float)
  p05 = float(np.quantile(profits, 0.05))
  return {
    "scenario": scenario,
    "profit_mean_rub": float(np.mean(profits)),
    "profit_sd_rub": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
    "profit_p05_rub": p05,
    "profit_p50_rub": float(np.median(profits)),
    "profit_p95_rub": float(np.quantile(profits, 0.95)),
    "cvar05_rub": float(np.mean(profits[profits <= p05]))
    if np.any(profits <= p05)
    else p05,
    "probability_loss_pct": float(np.mean(profits < 0) * 100.0),
    "probability_cash_insolvency_pct": float(frame["cash_insolvency"].mean() * 100.0),
    "minimum_cash_balance_p05_rub": float(
      frame["minimum_cash_balance_rub"].quantile(0.05)
    ),
    "avg_revenue_rub": float(frame["revenue_rub"].mean()),
    "avg_occupancy_pct": float(frame["occupancy_pct"].mean()),
    "avg_acceptance_pct": float(frame["acceptance_pct"].mean()),
    "avg_lost_requests": float(frame["lost_requests"].mean()),
    "avg_cleaner_utilization_pct": float(frame["cleaner_utilization_pct"].mean()),
    "avg_cleaning_wait_hours": float(frame["mean_cleaning_wait_hours"].mean()),
    "avg_cash_end_rub": float(frame["cash_end_rub"].mean()),
  }


# ---------------------------------------------------------------------------
# Validation, optimization and experiments
# ---------------------------------------------------------------------------


def run_self_checks(config: ModelConfig, market: MarketData) -> float:
  weekdays, stays, calendar = market.weekdays, market.stays, market.calendar
  assert weekdays["weekday_num"].tolist() == list(range(7))
  assert weekdays["weekday"].is_unique
  assert stays["bucket"].is_unique
  assert np.isclose(stays["share_pct"].sum(), 100.0, atol=1.0)

  rng = np.random.default_rng(98765)
  sample = np.array([_stay_sampler(stays, rng) for _ in range(50000)])
  expected_mean = float(
    np.average(
      [
        np.dot(
          np.asarray(row.model_values, dtype=float),
          np.asarray(row.model_probabilities, dtype=float),
        )
        for row in stays.itertuples(index=False)
      ],
      weights=stays["share_pct"].to_numpy(dtype=float),
    )
  )
  if abs(float(sample.mean()) - expected_mean) > 0.06:
    raise AssertionError(
      f"Sampled stay mean {sample.mean():.3f} differs from calibrated mean {expected_mean:.3f}"
    )

  result = simulate_once(
    config, Decision(), weekdays, stays, calendar, seed=config.random_seed
  )
  if result["requests"] != result["accepted_bookings"] + result["lost_requests"]:
    raise AssertionError("Request accounting identity failed")
  if not 0.0 <= result["occupancy_pct"] <= 100.0:
    raise AssertionError("Occupancy is outside [0, 100]")
  if not math.isclose(
    result["profit_rub"],
    result["revenue_rub"] - result["total_costs_rub"],
    abs_tol=1e-6,
  ):
    raise AssertionError("Profit identity failed")
  if not math.isclose(
    result["cash_end_rub"], config.opening_cash_rub + result["profit_rub"], abs_tol=1e-5
  ):
    raise AssertionError("Cash-flow identity failed")
  return expected_mean


def optimize_decision(
  config: ModelConfig, market: MarketData
) -> tuple[Decision, dict[str, Any]]:
  seeds = list(range(10000, 10000 + config.optimization_runs))
  candidates: list[dict[str, Any]] = []

  for cleaner_count in (1, 2, 3):

    def objective(cleaner_count: int, values: np.ndarray) -> float:
      decision = Decision(float(values[0]), float(values[1]), cleaner_count)
      runs = run_replications(
        config, decision, market.weekdays, market.stays, market.calendar, seeds
      )
      profits = runs["profit_rub"].to_numpy(dtype=float)
      std = float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0
      return -float(profits.mean() - config.risk_aversion * std)

    result = differential_evolution(
      partial(lambda x, i: objective(i, x), i=cleaner_count),
      bounds=[(0.80, 1.25), (0.0, 0.30)],
      seed=config.random_seed + cleaner_count,
      maxiter=config.optimizer_maxiter,
      popsize=config.optimizer_popsize,
      tol=0.015,
      polish=False,
      updating="immediate",
      workers=1,
    )
    candidates.append({
      "decision": Decision(float(result.x[0]), float(result.x[1]), cleaner_count),
      "objective": float(-result.fun),
      "success": bool(result.success),
      "message": str(result.message),
      "function_evaluations": int(result.nfev),
    })

  best = max(candidates, key=lambda item: item["objective"])
  info = {
    "method": "SciPy differential_evolution; cleaner count enumerated as 1, 2, 3",
    "objective": f"maximize mean(profit) - {config.risk_aversion} * sample_std(profit)",
    "optimization_replications_per_evaluation": len(seeds),
    "best_decision": {
      "price_multiplier": best["decision"].price_multiplier,
      "weekend_markup": best["decision"].weekend_markup,
      "cleaners": best["decision"].cleaners,
      "estimated_risk_adjusted_profit_rub": best["objective"],
    },
    "candidate_diagnostics": [
      {
        "cleaners": item["decision"].cleaners,
        "price_multiplier": item["decision"].price_multiplier,
        "weekend_markup": item["decision"].weekend_markup,
        "objective": item["objective"],
        "success": item["success"],
        "message": item["message"],
        "function_evaluations": item["function_evaluations"],
      }
      for item in candidates
    ],
  }
  return best["decision"], info


def run_scenarios(
  config: ModelConfig, market: MarketData, optimized: Decision, results_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
  decisions = {
    "baseline": Decision(1.00, 0.00, 1),
    "weekend_markup_10pct": Decision(1.00, 0.10, 1),
    "two_cleaners": Decision(1.00, 0.10, 2),
    "lower_price_90pct": Decision(0.90, 0.10, 1),
    "higher_price_110pct": Decision(1.10, 0.10, 1),
    "optimized": optimized,
  }
  seeds = list(range(20000, 20000 + config.monte_carlo_runs))
  all_runs: dict[str, pd.DataFrame] = {}
  summaries: list[dict[str, Any]] = []

  for name, decision in decisions.items():
    frame = run_replications(
      config, decision, market.weekdays, market.stays, market.calendar, seeds
    )
    all_runs[name] = frame
    frame.to_csv(results_dir / f"runs_{name}.csv", index=False)
    summaries.append(summarize(frame, name))
    print(f"Scenario complete: {name}", flush=True)

  stress = run_replications(
    config,
    decisions["baseline"],
    market.weekdays,
    market.stays,
    market.calendar,
    seeds,
    demand_scale=0.80,
    cost_scale=1.10,
    cleaning_time_scale=1.25,
    owner_block_share=0.15,
  )
  stress.to_csv(results_dir / "runs_stress.csv", index=False)
  summaries.append(summarize(stress, "stress"))

  summary = pd.DataFrame(summaries)
  summary.to_csv(results_dir / "scenario_summary.csv", index=False)
  return all_runs, stress, summary


def run_sensitivity(
  config: ModelConfig, market: MarketData, results_dir: Path
) -> pd.DataFrame:
  seeds = list(range(30000, 30000 + config.sensitivity_runs))
  sensitivity_rows: list[dict[str, float]] = []
  for elasticity in (0.6, 1.2, 2.0):
    for multiplier in (0.80, 0.90, 1.00, 1.10, 1.20):
      decision = Decision(multiplier, 0.10, 1)
      frame = run_replications(
        config,
        decision,
        market.weekdays,
        market.stays,
        market.calendar,
        seeds,
        price_elasticity=elasticity,
      )
      sensitivity_rows.append({
        "price_elasticity_assumption": elasticity,
        "price_multiplier": multiplier,
        "mean_profit_rub": float(frame["profit_rub"].mean()),
        "profit_p05_rub": float(frame["profit_rub"].quantile(0.05)),
        "mean_occupancy_pct": float(frame["occupancy_pct"].mean()),
        "mean_lost_requests": float(frame["lost_requests"].mean()),
      })

  sensitivity = pd.DataFrame(sensitivity_rows)
  sensitivity.to_csv(results_dir / "price_elasticity_sensitivity.csv", index=False)

  block_rows: list[dict[str, float]] = []
  for block_share in (0.0, 0.10, 0.20):
    frame = run_replications(
      config,
      Decision(),
      market.weekdays,
      market.stays,
      market.calendar,
      seeds,
      owner_block_share=block_share,
    )
    block_rows.append({
      "owner_block_share_assumption": block_share,
      "mean_profit_rub": float(frame["profit_rub"].mean()),
      "mean_occupancy_pct": float(frame["occupancy_pct"].mean()),
      "mean_lost_requests": float(frame["lost_requests"].mean()),
      "mean_acceptance_pct": float(frame["acceptance_pct"].mean()),
    })
  pd.DataFrame(block_rows).to_csv(
    results_dir / "owner_blocks_sensitivity.csv", index=False
  )
  return sensitivity


def run_dynamic_analysis(
  config: ModelConfig, market: MarketData, optimized: Decision, results_dir: Path
) -> pd.DataFrame:
  baseline_revenue: list[np.ndarray] = []
  optimized_revenue: list[np.ndarray] = []
  baseline_occupancy: list[np.ndarray] = []
  optimized_occupancy: list[np.ndarray] = []
  baseline_cash: list[np.ndarray] = []
  optimized_cash: list[np.ndarray] = []

  for index in range(config.dynamic_runs):
    seed = 40000 + index
    base = simulate_once(
      config, Decision(), market.weekdays, market.stays, market.calendar, seed
    )
    opt = simulate_once(
      config, optimized, market.weekdays, market.stays, market.calendar, seed
    )
    baseline_revenue.append(base["daily_revenue"])
    baseline_occupancy.append(base["daily_occupancy"] / config.units * 100.0)
    baseline_cash.append(base["daily_cash_balance"])
    optimized_revenue.append(opt["daily_revenue"])
    optimized_occupancy.append(opt["daily_occupancy"] / config.units * 100.0)
    optimized_cash.append(opt["daily_cash_balance"])

  base_rev = np.asarray(baseline_revenue)
  opt_rev = np.asarray(optimized_revenue)
  base_occ = np.asarray(baseline_occupancy)
  opt_occ = np.asarray(optimized_occupancy)
  base_cash = np.asarray(baseline_cash)
  opt_cash = np.asarray(optimized_cash)
  dynamics = pd.DataFrame({
    "day": np.arange(1, config.horizon_days + 1),
    "date": [d.isoformat() for d in market.calendar["date"]],
    "baseline_revenue_mean_rub": base_rev.mean(axis=0),
    "baseline_revenue_p10_rub": np.quantile(base_rev, 0.10, axis=0),
    "baseline_revenue_p90_rub": np.quantile(base_rev, 0.90, axis=0),
    "optimized_revenue_mean_rub": opt_rev.mean(axis=0),
    "optimized_revenue_p10_rub": np.quantile(opt_rev, 0.10, axis=0),
    "optimized_revenue_p90_rub": np.quantile(opt_rev, 0.90, axis=0),
    "baseline_occupancy_mean_pct": base_occ.mean(axis=0),
    "optimized_occupancy_mean_pct": opt_occ.mean(axis=0),
    "baseline_cash_mean_rub": base_cash.mean(axis=0),
    "baseline_cash_p10_rub": np.quantile(base_cash, 0.10, axis=0),
    "baseline_cash_p90_rub": np.quantile(base_cash, 0.90, axis=0),
    "optimized_cash_mean_rub": opt_cash.mean(axis=0),
    "optimized_cash_p10_rub": np.quantile(opt_cash, 0.10, axis=0),
    "optimized_cash_p90_rub": np.quantile(opt_cash, 0.90, axis=0),
  })
  dynamics.to_csv(results_dir / "daily_dynamics.csv", index=False)
  return dynamics


# ---------------------------------------------------------------------------
# Figures and result metadata
# ---------------------------------------------------------------------------


def create_figures(
  all_runs: dict[str, pd.DataFrame],
  stress: pd.DataFrame,
  dynamics: pd.DataFrame,
  sensitivity: pd.DataFrame,
  figures_dir: Path,
) -> None:
  scenario_names = list(all_runs) + ["stress"]
  profit_data = [all_runs[name]["profit_rub"].to_numpy() for name in all_runs]
  profit_data.append(stress["profit_rub"].to_numpy())

  fig, ax = plt.subplots(figsize=(11, 6))
  ax.boxplot(profit_data, tick_labels=scenario_names, showfliers=False)
  ax.set_title("90-day net profit: Monte Carlo distribution")
  ax.set_ylabel("Net profit, RUB")
  ax.tick_params(axis="x", rotation=25)
  ax.grid(axis="y", alpha=0.25)
  fig.tight_layout()
  fig.savefig(figures_dir / "scenario_profit_boxplot.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(10, 6))
  for name, frame in (
    ("baseline", all_runs["baseline"]),
    ("optimized", all_runs["optimized"]),
    ("stress", stress),
  ):
    ax.hist(frame["profit_rub"], bins=35, alpha=0.45, density=True, label=name)
  ax.set_title("Distribution of 90-day net profit")
  ax.set_xlabel("Net profit, RUB")
  ax.set_ylabel("Density")
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(figures_dir / "profit_distribution.png", dpi=180)
  plt.close(fig)

  x = dynamics["day"].to_numpy()
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, dynamics["baseline_revenue_mean_rub"], label="Baseline")
  ax.fill_between(
    x,
    dynamics["baseline_revenue_p10_rub"],
    dynamics["baseline_revenue_p90_rub"],
    alpha=0.15,
  )
  ax.plot(x, dynamics["optimized_revenue_mean_rub"], label="Optimized")
  ax.fill_between(
    x,
    dynamics["optimized_revenue_p10_rub"],
    dynamics["optimized_revenue_p90_rub"],
    alpha=0.15,
  )
  ax.set_title("Daily earned revenue: mean and 10-90% band")
  ax.set_xlabel("Simulation day")
  ax.set_ylabel("Revenue, RUB/day")
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(figures_dir / "daily_revenue_dynamics.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, dynamics["baseline_occupancy_mean_pct"], label="Baseline")
  ax.plot(x, dynamics["optimized_occupancy_mean_pct"], label="Optimized")
  ax.set_title("Daily portfolio occupancy")
  ax.set_xlabel("Simulation day")
  ax.set_ylabel("Occupied units, %")
  ax.set_ylim(0, 100)
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(figures_dir / "daily_occupancy_dynamics.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(9, 6))
  for elasticity, part in sensitivity.groupby("price_elasticity_assumption"):
    part = part.sort_values("price_multiplier")
    ax.plot(
      part["price_multiplier"],
      part["mean_profit_rub"],
      marker="o",
      label=f"Elasticity = {elasticity:.1f}",
    )
  ax.set_title("Sensitivity of mean profit to price policy")
  ax.set_xlabel("Price multiplier relative to calendar market price")
  ax.set_ylabel("Mean net profit, RUB / horizon")
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(figures_dir / "price_elasticity_sensitivity.png", dpi=180)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(x, dynamics["baseline_cash_mean_rub"], label="Baseline")
  ax.fill_between(
    x, dynamics["baseline_cash_p10_rub"], dynamics["baseline_cash_p90_rub"], alpha=0.15
  )
  ax.plot(x, dynamics["optimized_cash_mean_rub"], label="Optimized")
  ax.fill_between(
    x,
    dynamics["optimized_cash_p10_rub"],
    dynamics["optimized_cash_p90_rub"],
    alpha=0.15,
  )
  ax.axhline(0, linewidth=1)
  ax.set_title("Daily cash balance: mean and 10-90% band")
  ax.set_xlabel("Simulation day")
  ax.set_ylabel("Cash balance, RUB")
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(figures_dir / "cash_balance_dynamics.png", dpi=180)
  plt.close(fig)


def write_market_diagnostics(
  config: ModelConfig,
  market: MarketData,
  results_dir: Path,
  stay_mean: float,
  stay_decay: float,
) -> None:
  market.monthly_summary.to_csv(results_dir / "market_monthly_summary.csv", index=False)
  calendar_export = market.calendar.copy()
  calendar_export["date"] = calendar_export["date"].map(date.isoformat)
  cols = [
    column
    for column in (
      "date",
      "weekday",
      "price_rub",
      "occupancy_pct",
      "usual_occupancy_pct",
      "delta_pp",
      "judged",
      "special",
      "verdict",
      "usual_price_rub",
      "price_delta_pct",
      "snapshot_date",
    )
    if column in calendar_export.columns
  ]
  calendar_export[cols].to_csv(results_dir / "market_calendar_inputs.csv", index=False)

  latest_month = market.monthly_summary.dropna(
    subset=["adr_rub", "occupancy_pct"]
  ).iloc[-1]
  diagnostics = {
    "city_id": market.city_id,
    "snapshot_date": config.snapshot_date,
    "calibration_month": str(latest_month["month"]),
    "source_adr_rub_latest_month_unweighted_district_mean": config.source_adr_rub,
    "source_occupancy_pct_latest_month_unweighted_district_mean": config.source_occupancy_pct,
    "source_districts_with_adr": int(latest_month["adr_districts"]),
    "source_districts_with_occupancy": int(latest_month["occupancy_districts"]),
    "source_listings_count": (
      None if pd.isna(latest_month["listings"]) else int(latest_month["listings"])
    ),
    "calendar_start": market.calendar["date"].iloc[0].isoformat(),
    "calendar_end": market.calendar["date"].iloc[-1].isoformat(),
    "calendar_price_mean_rub": float(market.calendar["price_rub"].mean()),
    "calendar_occupancy_pace_mean_pct": float(market.calendar["occupancy_pct"].mean()),
    "calendar_occupancy_pace_min_pct": float(market.calendar["occupancy_pct"].min()),
    "calendar_occupancy_pace_max_pct": float(market.calendar["occupancy_pct"].max()),
    "calendar_occupancy_interpretation": (
      "Observed calendar booking/closure pace at the snapshot date; not treated as final occupancy"
    ),
    "stay_length_model_mean_nights": stay_mean,
    "stay_length_decay_parameter": stay_decay,
    "stay_length_tail_cap_nights": config.stay_tail_cap_nights,
    "stay_length_assumption": (
      "Discrete exponential weights within each interval bucket; open-ended bucket capped by config"
    ),
  }
  (results_dir / "market_data_diagnostics.json").write_text(
    json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
  )


def write_optimization_info(
  config: ModelConfig,
  market: MarketData,
  optimization_info: dict[str, Any],
  results_dir: Path,
) -> None:
  optimization_info["calibration"] = {
    "source_adr_rub": config.source_adr_rub,
    "source_occupancy_pct": config.source_occupancy_pct,
    "start_date": config.start_date,
    "horizon_days": config.horizon_days,
    "market_calendar_price_mean_rub": float(market.calendar["price_rub"].mean()),
  }
  optimization_info["replications_final"] = config.monte_carlo_runs
  optimization_info["seed_start_final_validation"] = 20000
  (results_dir / "optimization.json").write_text(
    json.dumps(optimization_info, ensure_ascii=False, indent=2), encoding="utf-8"
  )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
  args = parse_args()
  config_path = args.config.expanduser().resolve()
  data_dir = args.data_dir.expanduser().resolve()
  if not config_path.is_file():
    raise FileNotFoundError(f"Model config not found: {config_path}")
  if not data_dir.is_dir():
    raise NotADirectoryError(f"Data directory not found: {data_dir}")

  config = ModelConfig.from_json(config_path)
  config.validate()
  results_dir, figures_dir = prepare_output_dirs(args.output_dir)

  market = read_market_data(data_dir, config)
  sampled_stay_mean = run_self_checks(config, market)
  write_market_diagnostics(
    config, market, results_dir, market.stay_mean, market.stay_decay
  )

  print(
    "Input data validated: "
    f"city={market.city_id}, period={config.start_date}..{market.calendar['date'].iloc[-1]}, "
    f"ADR={config.source_adr_rub:,.2f} RUB, "
    f"occupancy={config.source_occupancy_pct:.2f}%, "
    f"mean_stay={sampled_stay_mean:.3f} nights",
    flush=True,
  )

  optimized, optimization_info = optimize_decision(config, market)
  print("Optimization complete", flush=True)

  all_runs, stress, summary = run_scenarios(config, market, optimized, results_dir)
  print("Scenario experiments complete", flush=True)

  sensitivity = run_sensitivity(config, market, results_dir)
  print("Sensitivity analysis complete", flush=True)

  dynamics = run_dynamic_analysis(config, market, optimized, results_dir)
  print("Dynamic trajectories complete", flush=True)

  create_figures(all_runs, stress, dynamics, sensitivity, figures_dir)
  write_optimization_info(config, market, optimization_info, results_dir)

  print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
  print(
    "\nOPTIMIZATION\n" + json.dumps(optimization_info, ensure_ascii=False, indent=2)
  )
  print(f"\nResults: {results_dir}")
  print(f"Figures: {figures_dir}")


if __name__ == "__main__":
  main()
