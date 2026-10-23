from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera


TARGET_COLUMN = 'shares'
TIME_COLUMN = 'timedelta'
NON_FEATURE_COLUMNS = {'url', TARGET_COLUMN}
STRUCTURAL_REDUNDANCIES = {'LDA_04', 'weekday_is_sunday', 'is_weekend'}
LOG_SKEW_CANDIDATES = {
  'n_tokens_content',
  'num_hrefs',
  'num_self_hrefs',
  'num_imgs',
  'num_videos',
  'num_keywords',
  'kw_max_min',
  'kw_min_max',
  'kw_max_max',
  'kw_avg_max',
  'kw_max_avg',
  'kw_avg_avg',
  'self_reference_min_shares',
  'self_reference_max_shares',
  'self_reference_avg_sharess',
  'timedelta',
}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (0.0005, 0.001, 0.005, 0.01, 0.05)
ELASTIC_ALPHAS = (0.001, 0.01, 0.1)
ELASTIC_L1_RATIOS = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class Fold:
  """Индексы и временные границы фолда."""

  train_indices: np.ndarray
  validation_indices: np.ndarray
  train_last_timedelta: float
  validation_first_timedelta: float
  validation_last_timedelta: float


@dataclass(frozen=True)
class Candidate:
  """Параметры кандидата регрессионной модели."""

  kind: str
  profile: str
  alpha: float | None = None
  l1_ratio: float | None = None
  retransformation: str = 'direct'

  @property
  def label(self):
    """Сформировать имя модели."""
    suffix = {'base': 'base', 'log_features': 'log_features', 'vif_reduced': 'vif_reduced'}.get(self.profile, self.profile)
    response_suffix = f'_{self.retransformation}' if self.kind != 'OLS_linear' else ''
    if self.alpha is None:
      return f'{self.kind}{response_suffix}_{suffix}'
    if self.l1_ratio is None:
      return f'{self.kind}_alpha{self.alpha:g}{response_suffix}_{suffix}'
    return f'{self.kind}_alpha{self.alpha:g}_l1ratio{self.l1_ratio:g}{response_suffix}_{suffix}'

  def parameters(self):
    """Вернуть параметры модели."""
    return {
      'kind': self.kind,
      'profile': self.profile,
      'alpha': self.alpha,
      'l1_ratio': self.l1_ratio,
      'retransformation': self.retransformation,
    }


@dataclass
class FittedCandidate:
  """Обученная модель и параметры обратного преобразования."""

  candidate: Candidate
  pipeline: Pipeline
  smearing_factor: float
  feature_columns: list[str]
  log_features: list[str]


@dataclass
class Evaluation:
  """Метрики модели на временной проверке."""

  candidate: Candidate
  cv_mae: float
  cv_rmse: float
  cv_r2: float
  cv_rmsle: float
  fold_metrics: list[dict[str, float | int]]
  convergence_warnings: int = 0


def configure_logging(output_dir):
  """Настроить логирование в файл и консоль."""
  output_dir.mkdir(parents=True, exist_ok=True)
  logger = logging.getLogger('onp_regression')
  logger.setLevel(logging.INFO)
  logger.handlers.clear()
  formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
  file_handler = logging.FileHandler(output_dir / 'analysis.log', encoding='utf-8')
  file_handler.setFormatter(formatter)
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  return logger


def read_dataset(path, logger):
  """Загрузить, проверить и очистить набор данных."""
  if not path.is_file():
    raise FileNotFoundError(f'CSV file does not exist: {path}')
  try:
    frame = pd.read_csv(path, skipinitialspace=True)
  except pd.errors.EmptyDataError as exc:
    raise ValueError('CSV file is empty or has no header row.') from exc
  except (OSError, UnicodeError, pd.errors.ParserError) as exc:
    raise ValueError(f"Could not read CSV '{path}': {exc}") from exc
  if frame.empty or len(frame.columns) == 0:
    raise ValueError('CSV contains no observations or columns.')
  if frame.columns.duplicated().any():
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    raise ValueError(f'CSV has duplicate column names: {duplicates}')
  required = {TARGET_COLUMN, TIME_COLUMN}
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f'Required columns are missing: {missing}')
  original_rows, original_columns = frame.shape
  exact_duplicates = int(frame.duplicated().sum())
  logger.info('Input: %s', path)
  logger.info('Raw shape: %d rows x %d columns; exact duplicate rows=%d', original_rows, original_columns, exact_duplicates)
  frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors='coerce')
  frame[TIME_COLUMN] = pd.to_numeric(frame[TIME_COLUMN], errors='coerce')
  target = frame[TARGET_COLUMN].to_numpy(dtype=float)
  time = frame[TIME_COLUMN].to_numpy(dtype=float)
  valid_target_time = np.isfinite(target) & (target >= 0) & np.isfinite(time)
  excluded_target_time = int((~valid_target_time).sum())
  if excluded_target_time:
    logger.warning('Excluding %d rows with missing/non-finite time or missing/negative target.', excluded_target_time)
  frame = frame.loc[valid_target_time].copy()
  if frame.empty:
    raise ValueError('No rows remain after validating shares >= 0 and finite timedelta.')
  non_numeric = [
    column
    for column in frame.columns
    if column not in NON_FEATURE_COLUMNS and column != TIME_COLUMN and (not pd.api.types.is_numeric_dtype(frame[column]))
  ]
  for column in non_numeric:
    frame[column] = pd.to_numeric(frame[column], errors='coerce')
  bounded_ratio_columns = [column for column in ('n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens') if column in frame.columns]
  invalid_ratio_cells = pd.DataFrame(False, index=frame.index, columns=bounded_ratio_columns)
  for column in bounded_ratio_columns:
    invalid_ratio_cells[column] = frame[column].notna() & ~frame[column].between(0.0, 1.0)
    if bool(invalid_ratio_cells[column].any()):
      logger.warning(
        'Column %s has %d ratio value(s) outside [0, 1]; treating these cells as missing.', column, int(invalid_ratio_cells[column].sum())
      )
      frame.loc[invalid_ratio_cells[column], column] = np.nan
  if not invalid_ratio_cells.empty and bool(invalid_ratio_cells.any(axis=None).any()):
    frame['data_quality_anomaly_flag'] = invalid_ratio_cells.any(axis=1).astype(float)
    logger.warning('Added data_quality_anomaly_flag for %d affected row(s); observations retained.', int(invalid_ratio_cells.any(axis=1).sum()))
  lda_columns = [f'LDA_{index:02d}' for index in range(5) if f'LDA_{index:02d}' in frame.columns]
  if len(lda_columns) == 5:
    lda_complete = frame.loc[:, lda_columns].notna().all(axis=1)
    lda_sum = frame.loc[:, lda_columns].sum(axis=1, min_count=5)
    lda_zero = lda_complete & np.isclose(lda_sum.to_numpy(dtype=float), 0.0, atol=1e-10)
    if bool(lda_zero.any()):
      if 'data_quality_anomaly_flag' in frame.columns:
        frame.loc[lda_zero, 'data_quality_anomaly_flag'] = 1.0
      else:
        frame['data_quality_anomaly_flag'] = lda_zero.astype(float)
      frame.loc[lda_zero, lda_columns] = np.nan
      logger.warning(
        'Found %d all-zero LDA topic vector(s); topic values set to missing for fold-local median imputation and data_quality_anomaly_flag updated.',
        int(lda_zero.sum()),
      )
  numeric_predictors = [
    column for column in frame.columns if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
  ]
  dropped_all_missing = []
  for column in numeric_predictors:
    frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    if frame[column].notna().sum() == 0:
      dropped_all_missing.append(column)
  if dropped_all_missing:
    logger.warning('Dropping all-missing predictor columns: %s', dropped_all_missing)
    frame = frame.drop(columns=dropped_all_missing)
  frame = frame.sort_values(TIME_COLUMN, ascending=False, kind='stable').reset_index(drop=True)
  logger.info('Usable shape: %d rows x %d columns; excluded invalid target/time rows=%d', len(frame), frame.shape[1], excluded_target_time)
  logger.info('timedelta unique groups=%d; range=[%s, %s]', frame[TIME_COLUMN].nunique(), frame[TIME_COLUMN].min(), frame[TIME_COLUMN].max())
  return frame


def choose_group_boundary(frame, train_fraction):
  """Найти границу между целыми временными группами."""
  group_sizes = frame.groupby(TIME_COLUMN, sort=False).size().to_numpy()
  if len(group_sizes) < 8:
    raise ValueError('At least 8 distinct time groups are needed for grouped temporal validation.')
  cumulative = np.cumsum(group_sizes)
  target_rows = train_fraction * len(frame)
  boundary = int(np.argmin(np.abs(cumulative[:-1] - target_rows))) + 1
  return min(max(boundary, 1), len(group_sizes) - 1)


def chronological_train_test_split(frame):
  """Разделить данные по времени без пересечения групп."""
  groups = frame[TIME_COLUMN].drop_duplicates().to_numpy()
  boundary = choose_group_boundary(frame, 0.8)
  train_groups = set(groups[:boundary].tolist())
  train_mask = frame[TIME_COLUMN].isin(train_groups)
  train_frame = frame.loc[train_mask].copy().reset_index(drop=True)
  test_frame = frame.loc[~train_mask].copy().reset_index(drop=True)
  if train_frame.empty or test_frame.empty:
    raise ValueError('Chronological split produced an empty train or test partition.')
  if set(train_frame[TIME_COLUMN]).intersection(set(test_frame[TIME_COLUMN])):
    raise RuntimeError('Temporal leakage: a timedelta group appears in both train and test.')
  return (train_frame, test_frame)


def make_expanding_folds(frame, number_of_folds=4):
  """Создать фолды расширяющейся временной проверки."""
  groups = frame[TIME_COLUMN].drop_duplicates().to_numpy()
  minimum_initial = max(3, math.ceil(len(groups) * 0.6))
  if len(groups) < minimum_initial + number_of_folds:
    raise ValueError('Not enough distinct time groups for expanding-window CV.')
  boundaries = np.linspace(minimum_initial, len(groups), number_of_folds + 1, dtype=int)
  folds = []
  for fold_number in range(number_of_folds):
    start = int(boundaries[fold_number])
    end = int(boundaries[fold_number + 1])
    if start >= end:
      continue
    train_groups = set(groups[:start].tolist())
    validation_groups = set(groups[start:end].tolist())
    train_indices = np.flatnonzero(frame[TIME_COLUMN].isin(train_groups).to_numpy())
    validation_indices = np.flatnonzero(frame[TIME_COLUMN].isin(validation_groups).to_numpy())
    if len(train_indices) == 0 or len(validation_indices) == 0:
      continue
    folds.append(
      Fold(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_last_timedelta=float(frame.iloc[train_indices][TIME_COLUMN].min()),
        validation_first_timedelta=float(frame.iloc[validation_indices][TIME_COLUMN].max()),
        validation_last_timedelta=float(frame.iloc[validation_indices][TIME_COLUMN].min()),
      )
    )
  if len(folds) < 2:
    raise ValueError('Fewer than two valid chronological CV folds could be created.')
  for fold in folds:
    if fold.train_last_timedelta < fold.validation_first_timedelta:
      raise RuntimeError('Temporal order check failed for an expanding-window fold.')
  return folds


def select_feature_columns(frame, logger):
  """Выбрать числовые признаки и убрать структурные дубликаты."""
  numeric = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])]
  removed = sorted(STRUCTURAL_REDUNDANCIES.intersection(numeric))
  selected = [column for column in numeric if column not in STRUCTURAL_REDUNDANCIES]
  logger.info('Predictors selected: %d; structurally redundant predictors excluded: %s', len(selected), removed)
  logger.info('Retained timedelta as a known-at-publication temporal predictor.')
  return selected


def transform_features(frame, columns, profile, log_features):
  """Применить преобразования к выбранному профилю признаков."""
  transformed = frame.loc[:, list(columns)].copy()
  if profile == 'log_features':
    for column in log_features:
      if column in transformed.columns:
        transformed[column] = np.log1p(transformed[column])
  elif profile not in {'base', 'vif_reduced'}:
    raise ValueError(f'Unknown feature profile: {profile}')
  return transformed


def estimator_for(candidate):
  """Создать регрессор заданного типа."""
  if candidate.kind in {'OLS_linear', 'OLS_log1p'}:
    return LinearRegression()
  if candidate.kind == 'Ridge_log1p':
    if candidate.alpha is None:
      raise ValueError('Ridge candidate requires alpha.')
    return Ridge(alpha=candidate.alpha)
  if candidate.kind == 'Lasso_log1p':
    if candidate.alpha is None:
      raise ValueError('Lasso candidate requires alpha.')
    return Lasso(alpha=candidate.alpha, max_iter=12000, tol=0.0001, selection='cyclic')
  if candidate.kind == 'ElasticNet_log1p':
    if candidate.alpha is None or candidate.l1_ratio is None:
      raise ValueError('ElasticNet candidate requires alpha and l1_ratio.')
    return ElasticNet(alpha=candidate.alpha, l1_ratio=candidate.l1_ratio, max_iter=12000, tol=0.0001, selection='cyclic')
  raise ValueError(f'Unknown model kind: {candidate.kind}')


def make_pipeline(candidate):
  """Создать конвейер импутации, масштабирования и регрессии."""
  return Pipeline([
    ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
    ('scaler', StandardScaler()),
    ('model', estimator_for(candidate)),
  ])


def uses_log_target(candidate):
  """Проверить, логарифмируется ли целевая переменная."""
  return candidate.kind != 'OLS_linear'


def model_target(candidate, target):
  """Преобразовать целевую переменную для обучения."""
  return np.log1p(target) if uses_log_target(candidate) else target


def smearing_factor(y_log, fitted_log):
  """Вычислить поправку Дуана по остаткам обучения."""
  residuals = y_log - fitted_log
  if np.any(residuals > 40):
    residuals = np.minimum(residuals, 40)
  value = float(np.mean(np.exp(residuals)))
  if not np.isfinite(value) or value <= 0:
    raise FloatingPointError(f'Invalid smearing factor: {value}')
  return value


def restore_target_scale(candidate, predicted_model_scale, smear):
  """Вернуть прогноз в исходный масштаб."""
  if uses_log_target(candidate):
    retransformation_factor = smear if candidate.retransformation == 'smearing' else 1.0
    predicted = np.exp(np.clip(predicted_model_scale, -30, 30)) * retransformation_factor - 1.0
  else:
    predicted = predicted_model_scale.copy()
  return np.maximum(predicted, 0.0)


def prediction_metrics(actual, predicted):
  """Вычислить метрики качества прогноза."""
  actual_safe = np.asarray(actual, dtype=float)
  predicted_safe = np.maximum(np.asarray(predicted, dtype=float), 0.0)
  rmse = float(np.sqrt(mean_squared_error(actual_safe, predicted_safe)))
  rmsle = float(np.sqrt(np.mean((np.log1p(actual_safe) - np.log1p(predicted_safe)) ** 2)))
  return {
    'mae': float(mean_absolute_error(actual_safe, predicted_safe)),
    'rmse': rmse,
    'r2': float(r2_score(actual_safe, predicted_safe)),
    'rmsle': rmsle,
    'median_absolute_error': float(median_absolute_error(actual_safe, predicted_safe)),
  }


def fit_candidate(candidate, x_train, y_train, feature_columns, log_features, logger=None):
  """Обучить модель и вычислить smearing-поправку."""
  transformed_x = transform_features(x_train, feature_columns, candidate.profile, log_features)
  target_for_fit = model_target(candidate, y_train)
  pipeline = make_pipeline(candidate)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    pipeline.fit(transformed_x, target_for_fit)
  convergence = [
    warning for warning in caught if 'converg' in warning.message.__class__.__name__.lower() or 'converg' in str(warning.message).lower()
  ]
  if caught and logger is not None:
    for warning in caught:
      logger.warning('Fit warning for %s: %s', candidate.label, warning.message)
  smear = 1.0
  if uses_log_target(candidate) and candidate.retransformation == 'smearing':
    train_fit_log = pipeline.predict(transformed_x)
    smear = smearing_factor(target_for_fit, train_fit_log)
  if convergence and logger is not None:
    logger.warning('Candidate %s generated %d convergence warning(s).', candidate.label, len(convergence))
  return FittedCandidate(candidate, pipeline, smear, feature_columns.copy(), log_features.copy())


def score_candidate(candidate, x_development, y_development, folds, feature_columns, log_features, logger):
  """Оценить модель на временных фолдах."""
  all_actual = []
  all_prediction = []
  fold_metrics = []
  warning_count = 0
  for fold_number, fold in enumerate(folds, start=1):
    x_train = x_development.iloc[fold.train_indices]
    x_valid = x_development.iloc[fold.validation_indices]
    y_train = y_development[fold.train_indices]
    y_valid = y_development[fold.validation_indices]
    train_features = transform_features(x_train, feature_columns, candidate.profile, log_features)
    valid_features = transform_features(x_valid, feature_columns, candidate.profile, log_features)
    target_train = model_target(candidate, y_train)
    pipeline = make_pipeline(candidate)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      pipeline.fit(train_features, target_train)
    convergence = [
      warning for warning in caught if 'converg' in warning.message.__class__.__name__.lower() or 'converg' in str(warning.message).lower()
    ]
    warning_count += len(convergence)
    for warning in caught:
      logger.warning('CV fit warning (%s fold %d): %s', candidate.label, fold_number, warning.message)
    predictions_model = pipeline.predict(valid_features)
    smear = 1.0
    if uses_log_target(candidate) and candidate.retransformation == 'smearing':
      predictions_train_model = pipeline.predict(train_features)
      smear = smearing_factor(target_train, predictions_train_model)
    predictions = restore_target_scale(candidate, predictions_model, smear)
    fold_score = prediction_metrics(y_valid, predictions)
    fold_metrics.append({'fold': fold_number, 'n_train': len(y_train), 'n_validation': len(y_valid), **fold_score})
    all_actual.append(y_valid)
    all_prediction.append(predictions)
  pooled_actual = np.concatenate(all_actual)
  pooled_prediction = np.concatenate(all_prediction)
  pooled = prediction_metrics(pooled_actual, pooled_prediction)
  logger.info(
    'CV %s: MAE=%.3f RMSE=%.3f R2=%.5f RMSLE=%.5f (folds=%d)',
    candidate.label,
    pooled['mae'],
    pooled['rmse'],
    pooled['r2'],
    pooled['rmsle'],
    len(fold_metrics),
  )
  return Evaluation(candidate, pooled['mae'], pooled['rmse'], pooled['r2'], pooled['rmsle'], fold_metrics, warning_count)


def parameter_grid(kind):
  """Подготовить сетку гиперпараметров."""
  if kind == 'OLS_linear':
    return [{'retransformation': 'none'}]
  retransformation_modes = ('direct', 'smearing')
  if kind == 'OLS_log1p':
    return [{'retransformation': mode} for mode in retransformation_modes]
  if kind == 'Ridge_log1p':
    return [{'alpha': alpha, 'retransformation': mode} for alpha in RIDGE_ALPHAS for mode in retransformation_modes]
  if kind == 'Lasso_log1p':
    return [{'alpha': alpha, 'retransformation': mode} for alpha in LASSO_ALPHAS for mode in retransformation_modes]
  if kind == 'ElasticNet_log1p':
    return [
      {'alpha': alpha, 'l1_ratio': ratio, 'retransformation': mode}
      for alpha in ELASTIC_ALPHAS
      for ratio in ELASTIC_L1_RATIOS
      for mode in retransformation_modes
    ]
  raise ValueError(f'No parameter grid for {kind}')


def tune_family(kind, profile, x_development, y_development, folds, feature_columns, log_features, logger):
  """Выбрать параметры модели по временной кросс-валидации."""
  evaluations = []
  for params in parameter_grid(kind):
    candidate = Candidate(kind=kind, profile=profile, **params)
    try:
      evaluation = score_candidate(candidate, x_development, y_development, folds, feature_columns, log_features, logger)
      evaluations.append(evaluation)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
      logger.error('Candidate failed (%s): %s', candidate.label, exc)
  if not evaluations:
    raise RuntimeError(f'Every candidate failed for model family {kind} and profile {profile}.')
  best = min(evaluations, key=lambda item: (item.cv_mae, item.cv_rmsle))
  logger.info('Best %s / %s: %s (CV MAE=%.3f)', kind, profile, best.candidate.label, best.cv_mae)
  return (best, evaluations)


def final_test_predictions(evaluation, development, test, feature_columns, log_features, logger):
  """Обучить выбранную модель и оценить на тестовом периоде."""
  x_train = development.loc[:, feature_columns]
  x_test = test.loc[:, feature_columns]
  y_train = development[TARGET_COLUMN].to_numpy(dtype=float)
  y_test = test[TARGET_COLUMN].to_numpy(dtype=float)
  fitted = fit_candidate(evaluation.candidate, x_train, y_train, feature_columns, log_features, logger)
  x_test_profile = transform_features(x_test, feature_columns, evaluation.candidate.profile, log_features)
  pred_model_scale = fitted.pipeline.predict(x_test_profile)
  predictions = restore_target_scale(evaluation.candidate, pred_model_scale, fitted.smearing_factor)
  metrics = prediction_metrics(y_test, predictions)
  retransformation_factor = fitted.smearing_factor if evaluation.candidate.retransformation == 'smearing' else 1.0
  raw_prediction = (
    pred_model_scale if not uses_log_target(evaluation.candidate) else np.exp(np.clip(pred_model_scale, -30, 30)) * retransformation_factor - 1
  )
  negative_rate = float(np.mean(np.asarray(raw_prediction) < 0))
  logger.info(
    'TEST %s: MAE=%.3f RMSE=%.3f R2=%.5f RMSLE=%.5f',
    evaluation.candidate.label,
    metrics['mae'],
    metrics['rmse'],
    metrics['r2'],
    metrics['rmsle'],
  )
  return (fitted, predictions, metrics, negative_rate)


def calculate_vif_and_condition(x_scaled, names):
  """Вычислить VIF, ранг матрицы и число обусловленности."""
  matrix = np.asarray(x_scaled, dtype=float)
  nonconstant = np.std(matrix, axis=0) > 1e-12
  matrix_nc = matrix[:, nonconstant]
  names_nc = [name for name, keep in zip(names, nonconstant, strict=True) if keep]
  if matrix_nc.shape[1] == 0:
    return (float('nan'), 0, float('inf'), [])
  centered = matrix_nc - matrix_nc.mean(axis=0, keepdims=True)
  scales = np.sqrt(np.sum(centered**2, axis=0))
  standardized = centered / np.where(scales > 0, scales, 1.0)
  correlation = standardized.T @ standardized
  try:
    inverse_corr = np.linalg.inv(correlation)
  except np.linalg.LinAlgError:
    inverse_corr = np.linalg.pinv(correlation, rcond=1e-12)
  vif_values = np.diag(inverse_corr)
  table = [{'feature': name, 'vif': float(value)} for name, value in zip(names_nc, vif_values, strict=True)]
  table.sort(key=lambda record: float(record['vif']), reverse=True)
  design = sm.add_constant(matrix_nc, has_constant='add')
  rank = int(np.linalg.matrix_rank(design))
  condition = float(np.linalg.cond(design))
  max_vif = float(max(vif_values)) if len(vif_values) else float('nan')
  return (max_vif, rank, condition, table)


def select_low_vif_features(frame, feature_columns, vif_threshold, logger):
  """Последовательно удалить признаки с высоким VIF."""
  if vif_threshold <= 1.0:
    raise ValueError('VIF threshold must be greater than 1.')
  raw = frame.loc[:, feature_columns]
  imputer = SimpleImputer(strategy='median', keep_empty_features=True)
  scaler = StandardScaler()
  matrix = scaler.fit_transform(imputer.fit_transform(raw))
  kept = feature_columns.copy()
  removed = []
  while len(kept) > 8:
    max_vif, _, _, vif_rows = calculate_vif_and_condition(matrix, kept)
    if not np.isfinite(max_vif) or max_vif <= vif_threshold or (not vif_rows):
      break
    worst = vif_rows[0]
    worst_name = str(worst['feature'])
    removed.append({'feature': worst_name, 'vif_at_removal': float(worst['vif'])})
    position = kept.index(worst_name)
    kept.pop(position)
    matrix = np.delete(matrix, position, axis=1)
  max_vif, _, condition, _ = calculate_vif_and_condition(matrix, kept)
  logger.info('VIF-reduced profile: kept=%d, removed=%d, final max VIF=%.3f, condition=%.3f', len(kept), len(removed), max_vif, condition)
  for item in removed:
    logger.info('VIF reduction removed %s at VIF %.3f', item['feature'], item['vif_at_removal'])
  return (kept, removed)


def run_diagnostics(fitted, development, feature_columns, logger):
  """Выполнить статистическую диагностику модели и остатков."""
  candidate = fitted.candidate
  x = transform_features(development[feature_columns], feature_columns, candidate.profile, fitted.log_features)
  x = fitted.pipeline.named_steps['scaler'].transform(fitted.pipeline.named_steps['imputer'].transform(x))
  design = sm.add_constant(x, has_constant='add')
  y = model_target(candidate, development[TARGET_COLUMN].to_numpy(float))
  model = sm.OLS(y, design, hasconst=True).fit()
  residuals = np.asarray(model.resid, dtype=float)
  max_vif, rank, condition, vif = calculate_vif_and_condition(x, feature_columns)
  d = {
    'model': candidate.label,
    'n_train': len(development),
    'n_features': x.shape[1],
    'design_rank': rank,
    'design_columns_with_intercept': design.shape[1],
    'full_rank': rank == design.shape[1],
    'condition_number': condition,
    'max_vif': max_vif,
    'vif_over_5': sum(r['vif'] > 5 for r in vif),
    'vif_over_10': sum(r['vif'] > 10 for r in vif),
    'ols_reference_r2_on_model_scale': float(model.rsquared),
    'ols_reference_f_pvalue': float(model.f_pvalue) if np.isfinite(model.f_pvalue) else None,
  }
  try:
    bp = het_breuschpagan(residuals, design)
    d.update(breusch_pagan_lm=float(bp[0]), breusch_pagan_pvalue=float(bp[1]))
  except (ValueError, np.linalg.LinAlgError) as exc:
    d['breusch_pagan_pvalue'] = None
    logger.warning('Тест Бройша—Пагана недоступен: %s', exc)
  try:
    reset = linear_reset(model, power=3, use_f=True)
    d.update(ramsey_reset_f=float(np.asarray(reset.fvalue).item()), ramsey_reset_pvalue=float(np.asarray(reset.pvalue).item()))
  except (ValueError, np.linalg.LinAlgError, AttributeError) as exc:
    d['ramsey_reset_pvalue'] = None
    logger.warning('Тест RESET недоступен: %s', exc)
  jb = jarque_bera(residuals)
  d.update(jarque_bera_statistic=float(jb[0]), jarque_bera_pvalue=float(jb[1]))
  daily = pd.DataFrame({TIME_COLUMN: development[TIME_COLUMN].to_numpy(), 'residual': residuals}).groupby(TIME_COLUMN)['residual'].mean()
  daily = daily.sort_index(ascending=False)
  d['daily_residual_group_count'] = len(daily)
  d['daily_mean_residual_durbin_watson'] = float(durbin_watson(daily)) if len(daily) >= 12 else None
  d['daily_mean_residual_ljung_box_pvalue'] = None
  if len(daily) >= 12:
    lag = min(10, max(1, len(daily) // 5))
    try:
      lb = sm.stats.acorr_ljungbox(daily, lags=[lag], return_df=True)
      d['daily_mean_residual_ljung_box_lag'] = lag
      d['daily_mean_residual_ljung_box_pvalue'] = float(lb['lb_pvalue'].iloc[0])
    except (ValueError, np.linalg.LinAlgError) as exc:
      logger.warning('Тест Льюнга—Бокса недоступен: %s', exc)
  logger.info('Диагностика %s: %s', candidate.label, d)
  for row in vif:
    logger.info('VIF %-32s %.4g', row['feature'], row['vif'])
  logger.info('Сводка OLS для диагностики:\n%s', model.summary().as_text())
  return d


def evaluate_constant_baselines(development, test, folds):
  """Оценить базовые прогнозы по среднему и медиане."""
  results = {}
  for strategy in ('median', 'mean'):
    pooled_actual = []
    pooled_predictions = []
    fold_metrics = []
    for fold_number, fold in enumerate(folds, start=1):
      fold_train_y = development.iloc[fold.train_indices][TARGET_COLUMN].to_numpy(dtype=float)
      fold_valid_y = development.iloc[fold.validation_indices][TARGET_COLUMN].to_numpy(dtype=float)
      constant = float(np.median(fold_train_y) if strategy == 'median' else np.mean(fold_train_y))
      prediction = np.full(len(fold_valid_y), constant, dtype=float)
      fold_metrics.append({
        'fold': fold_number,
        'n_train': len(fold_train_y),
        'n_validation': len(fold_valid_y),
        **prediction_metrics(fold_valid_y, prediction),
      })
      pooled_actual.append(fold_valid_y)
      pooled_predictions.append(prediction)
    cv_metrics = prediction_metrics(np.concatenate(pooled_actual), np.concatenate(pooled_predictions))
    dev_y = development[TARGET_COLUMN].to_numpy(dtype=float)
    test_y = test[TARGET_COLUMN].to_numpy(dtype=float)
    constant = float(np.median(dev_y) if strategy == 'median' else np.mean(dev_y))
    test_metrics = prediction_metrics(test_y, np.full(len(test_y), constant, dtype=float))
    name = f'Baseline_{strategy}'
    results[name] = {'name': name, 'constant': constant, 'cv_metrics': cv_metrics, 'test_metrics': test_metrics, 'fold_metrics': fold_metrics}
  return results


def run_analysis(input_path, output_dir):
  """Выполнить исследование и сохранить лог с прогнозами."""
  logger = configure_logging(output_dir)
  start = pd.Timestamp.now(tz='UTC')
  logger.info('Начало анализа: %s', start.isoformat())
  frame = read_dataset(input_path, logger)
  features = select_feature_columns(frame, logger)
  quality = frame[features]
  logger.info(
    'Качество данных: строк=%d, столбцов=%d, группы времени=%d, дубликаты=%d',
    len(frame),
    frame.shape[1],
    frame[TIME_COLUMN].nunique(),
    int(frame.duplicated().sum()),
  )
  logger.info(
    'Пропуски признаков=%d, бесконечные значения=%d, аномальные строки=%d',
    int(quality.isna().sum().sum()),
    int(np.isinf(quality.to_numpy(float)).sum()),
    int(frame.get('data_quality_anomaly_flag', pd.Series(0, index=frame.index)).sum()),
  )
  logger.info(
    'Целевая переменная shares: mean=%.4g median=%.4g max=%.4g skew=%.4g',
    frame[TARGET_COLUMN].mean(),
    frame[TARGET_COLUMN].median(),
    frame[TARGET_COLUMN].max(),
    frame[TARGET_COLUMN].skew(),
  )
  train, test = chronological_train_test_split(frame)
  logger.info(
    'Хронологическое разделение: train=%d (%.2f%%), test=%d (%.2f%%)',
    len(train),
    len(train) / len(frame) * 100,
    len(test),
    len(test) / len(frame) * 100,
  )
  logger.info('Группы времени: train=%d, test=%d', train[TIME_COLUMN].nunique(), test[TIME_COLUMN].nunique())
  log_features = sorted(LOG_SKEW_CANDIDATES.intersection(features))
  skipped = sorted({'kw_min_min', 'kw_avg_min', 'kw_min_avg'}.intersection(features))
  logger.info('Логарифмируемые признаки: %s; пропущенные из-за возможных отрицательных значений: %s', log_features, skipped)
  folds = make_expanding_folds(train, number_of_folds=3)
  for i, fold in enumerate(folds, 1):
    logger.info(
      'CV fold %d: train=%d, valid=%d, train time=%s, valid=[%s,%s]',
      i,
      len(fold.train_indices),
      len(fold.validation_indices),
      fold.train_last_timedelta,
      fold.validation_last_timedelta,
      fold.validation_first_timedelta,
    )
  x, y = (train[features].copy(), train[TARGET_COLUMN].to_numpy(float))
  reduced, removed = select_low_vif_features(x.iloc[folds[0].train_indices], features, 10.0, logger)
  profile_features = {'base': features, 'log_features': features, 'vif_reduced': reduced}
  kinds = ['OLS_linear', 'OLS_log1p', 'Ridge_log1p', 'Lasso_log1p', 'ElasticNet_log1p']
  best, evaluations = ([], [])
  for profile, columns in profile_features.items():
    for kind in kinds:
      winner, tested = tune_family(kind, profile, x, y, folds, columns, log_features, logger)
      best.append(winner)
      evaluations.extend(tested)
  by_profile = {p: [e for e in best if e.candidate.profile == p] for p in profile_features}
  base = min(by_profile['base'], key=lambda e: (e.cv_mae, e.cv_rmsle))
  transformed = min(by_profile['log_features'], key=lambda e: (e.cv_mae, e.cv_rmsle))
  vif_best = min(by_profile['vif_reduced'], key=lambda e: (e.cv_mae, e.cv_rmsle))
  final = min(best, key=lambda e: (e.cv_mae, e.cv_rmsle))
  same_log = next(e for e in by_profile['log_features'] if e.candidate.kind == base.candidate.kind)
  same_vif = next(e for e in by_profile['vif_reduced'] if e.candidate.kind == base.candidate.kind)
  for title, item in (
    ('Лучшая базовая', base),
    ('Лучшая с log-признаками', transformed),
    ('Лучшая после VIF', vif_best),
    ('Итоговая по CV', final),
  ):
    logger.info(
      '%s модель: %s; CV MAE=%.4f RMSE=%.4f R2=%.5f RMSLE=%.5f',
      title,
      item.candidate.label,
      item.cv_mae,
      item.cv_rmse,
      item.cv_r2,
      item.cv_rmsle,
    )
  for title, item in (('log-преобразование', same_log), ('VIF-сокращение', same_vif)):
    logger.info(
      'Эффект %s для %s: MAE %.4f -> %.4f (%+.2f%%)',
      title,
      base.candidate.kind,
      base.cv_mae,
      item.cv_mae,
      (item.cv_mae - base.cv_mae) / base.cv_mae * 100,
    )
  baselines = evaluate_constant_baselines(train, test, folds)
  for name, value in baselines.items():
    logger.info('%s: constant=%.4g; CV=%s; TEST=%s', name, value['constant'], value['cv_metrics'], value['test_metrics'])
  paired = []
  if uses_log_target(base.candidate):
    c = base.candidate
    mode = 'smearing' if c.retransformation == 'direct' else 'direct'
    alt = Candidate(c.kind, c.profile, c.alpha, c.l1_ratio, mode)
    other = next((e for e in evaluations if e.candidate == alt), None)
    if other:
      paired = [base, other]
      logger.info(
        'Сравнение обратного преобразования: %s MAE=%.4f; %s MAE=%.4f', base.candidate.label, base.cv_mae, other.candidate.label, other.cv_mae
      )
  shortlist = list({e.candidate: e for e in [*best, *paired]}.values())
  test_metrics, fitted_models, predictions, negative_rates = ({}, {}, {}, {})
  for evaluation in shortlist:
    label = evaluation.candidate.label
    fitted, pred, metrics, negative = final_test_predictions(
      evaluation, train, test, profile_features[evaluation.candidate.profile], log_features, logger
    )
    fitted_models[label], predictions[label] = (fitted, pred)
    test_metrics[label], negative_rates[label] = (metrics, negative)
    logger.info('TEST %s: %s; negative before clipping=%.3f%%', label, metrics, negative * 100)
  raw_candidate = Candidate('OLS_linear', 'base', retransformation='none')
  raw_fitted = fit_candidate(raw_candidate, train[features], y, features, log_features, logger)
  raw_diag = run_diagnostics(raw_fitted, train, features, logger)
  selected = fitted_models[final.candidate.label]
  diagnostic = run_diagnostics(selected, train, selected.feature_columns, logger)
  logger.info('VIF-сокращение: удалены признаки %s', removed)
  logger.info('Диагностика OLS(shares): %s', raw_diag)
  logger.info('Диагностика итоговой модели: %s', diagnostic)
  prediction_frame = test[[TIME_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: 'actual_shares'}).copy()
  for evaluation in shortlist:
    label = evaluation.candidate.label
    prediction_frame[f'pred_{label}'] = predictions[label]
    prediction_frame[f'residual_{label}'] = prediction_frame['actual_shares'] - predictions[label]
  for name, value in baselines.items():
    prediction_frame[f'pred_{name}'] = value['constant']
    prediction_frame[f'residual_{name}'] = prediction_frame['actual_shares'] - value['constant']
  prediction_path = output_dir / 'predictions.csv'
  prediction_frame.to_csv(prediction_path, index=False)
  median = baselines['Baseline_median']
  final_test_mae = test_metrics[final.candidate.label]['mae']
  cv_delta = (final.cv_mae - base.cv_mae) / base.cv_mae * 100 if base.cv_mae else 0.0
  test_delta = (
    (final_test_mae - test_metrics[base.candidate.label]['mae']) / test_metrics[base.candidate.label]['mae'] * 100
    if test_metrics[base.candidate.label]['mae']
    else 0.0
  )
  logger.info('Итог против базового победителя: CV MAE=%+.2f%%; TEST MAE=%+.2f%%', cv_delta, test_delta)
  logger.info(
    'Итог против медианного baseline: CV MAE=%+.2f%%; TEST MAE=%+.2f%%',
    (final.cv_mae - median['cv_metrics']['mae']) / median['cv_metrics']['mae'] * 100,
    (final_test_mae - median['test_metrics']['mae']) / median['test_metrics']['mae'] * 100,
  )
  logger.info('Параметры и CV-метрики всех кандидатов:')
  for evaluation in evaluations:
    logger.info(
      '%s: CV MAE=%.4f RMSE=%.4f R2=%.5f RMSLE=%.5f folds=%s warnings=%d',
      evaluation.candidate.label,
      evaluation.cv_mae,
      evaluation.cv_rmse,
      evaluation.cv_r2,
      evaluation.cv_rmsle,
      evaluation.fold_metrics,
      evaluation.convergence_warnings,
    )
  logger.info('Предсказания сохранены: %s', prediction_path)
  logger.info('Завершено: %s; длительность=%s', pd.Timestamp.now(tz='UTC').isoformat(), pd.Timestamp.now(tz='UTC') - start)
  print('\n=== Итог исследования ===')
  print(f'Строк: {len(frame):,}; признаков: {len(features)}; train: {len(train):,}; test: {len(test):,}')
  print(f"Лучшая базовая модель: {base.candidate.label}; CV MAE={base.cv_mae:.3f}; test MAE={test_metrics[base.candidate.label]['mae']:.3f}")
  print(f'Итоговая модель: {final.candidate.label}; CV MAE={final.cv_mae:.3f}; test MAE={final_test_mae:.3f}')
  print(f'Итог относительно базовой модели: CV MAE={cv_delta:+.2f}%; test MAE={test_delta:+.2f}%')
  print(
    f"Диагностика: rank={diagnostic['design_rank']}/{diagnostic['design_columns_with_intercept']}; "
    f"max VIF={diagnostic['max_vif']}; BP p={diagnostic.get('breusch_pagan_pvalue')}; RESET p={diagnostic.get('ramsey_reset_pvalue')}"
  )
  print(f"Прогнозы: {prediction_path}; журнал: {output_dir / 'analysis.log'}")
  return 0


def parse_arguments(arguments=None):
  """Разобрать аргументы командной строки."""
  parser = argparse.ArgumentParser(description='Сравнение регрессионных моделей с временной проверкой.')
  parser.add_argument('csv', type=Path, help='Путь к входному CSV-файлу')
  parser.add_argument('--output-dir', type=Path, default=Path('outputs'), help='Каталог для результатов (по умолчанию: outputs)')
  return parser.parse_args(arguments)


def main(arguments=None):
  """Запустить анализ и обработать ожидаемые ошибки."""
  args = parse_arguments(arguments)
  try:
    return run_analysis(args.csv, args.output_dir)
  except (OSError, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
    print(f'Ошибка исследования: {exc}', file=sys.stderr)
    return 2


if __name__ == '__main__':
  raise SystemExit(main())
