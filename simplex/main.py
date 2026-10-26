"""
[ММТС] ЛР №2: Решение задачи линейного программирования симплекс-методом

Зависимости:
- scipy>=1.18.1,<1.19.0
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import linprog


GROUPS = ("Озимые зерновые", "Масличные культуры", "Кормовые культуры")


@dataclass(frozen=True)
class Crop:
  group: str
  name: str
  area_2020_thousand_ha: float
  harvest_2020_thousand_t: float
  price_rub_per_t: float


# Коэффициенты перенесены из последней Excel/R-модели.
# Для цен и некоторых остаточных категорий используются принятые в исследовании
# условные значения; они считаются исходными предпосылками данной версии модели.
CROPS = (
  Crop(GROUPS[0], "Пшеница озимая", 16914, 63200, 12240),
  Crop(GROUPS[0], "Рожь озимая", 980, 2400, 9662),
  Crop(GROUPS[0], "Ячмень озимый", 731, 2300, 9742),
  Crop(GROUPS[0], "Тритикале", 98, 300, 11900),
  Crop(GROUPS[1], "Подсолнечник", 8545, 13300, 24540),
  Crop(GROUPS[1], "Лён-кудряш", 1030, 788, 30000),
  Crop(GROUPS[1], "Соя", 2858, 4308, 26859),
  Crop(GROUPS[1], "Горчица", 201, 103, 29000),
  Crop(GROUPS[1], "Рапс", 1488, 2572, 27570),
  Crop(GROUPS[1], "Прочие масличные культуры (остаток)", 276, 129, 30000),
  Crop(GROUPS[2], "Кормовые корнеплоды", 16, 400, 3000),
  Crop(GROUPS[2], "Кукуруза на корм", 1258, 24800, 1500),
  Crop(GROUPS[2], "Сено многолетних трав", 9927, 8300, 7500),
  Crop(GROUPS[2], "Сено однолетних трав", 3373, 2100, 7500),
  Crop(GROUPS[2], "Прочие кормовые культуры (остаток)", 177, 1770, 3000),
)


@dataclass(frozen=True)
class Scenario:
  number: int
  alpha: float
  beta: float
  gamma: float


def build_model_data() -> dict[str, Any]:
  """Рассчитывает производные коэффициенты, как в Excel.

  Excel-аналоги:
    q_i = E_i / D_i — модельная урожайность (столбец G);
    r_i = p_i * q_i — выручка с 1 га, руб./га (столбец I);
    c_i = r_i / 1_000_000 — коэффициент цели в млрд руб. на тыс. га.
  """
  names = [crop.name for crop in CROPS]
  groups = [crop.group for crop in CROPS]
  area = [float(crop.area_2020_thousand_ha) for crop in CROPS]
  harvest = [float(crop.harvest_2020_thousand_t) for crop in CROPS]
  price = [float(crop.price_rub_per_t) for crop in CROPS]
  yield_t_per_ha = [h / a for h, a in zip(harvest, area)]

  # При x_i в тыс. га величина p_i*q_i*x_i получается в тыс. руб.;
  # деление на 1e6 переводит её в млрд руб.
  revenue_coeff = [p * q / 1_000_000 for p, q in zip(price, yield_t_per_ha)]
  base_crop_revenue = [c * a for c, a in zip(revenue_coeff, area)]
  base_total_area = sum(area)

  group_indices = {
    group: [i for i, crop_group in enumerate(groups) if crop_group == group]
    for group in GROUPS
  }
  base_group_area = {
    group: sum(area[i] for i in idx) for group, idx in group_indices.items()
  }
  base_group_harvest = {
    group: sum(yield_t_per_ha[i] * area[i] for i in idx)
    for group, idx in group_indices.items()
  }
  base_group_revenue = {
    group: sum(base_crop_revenue[i] for i in idx)
    for group, idx in group_indices.items()
  }

  return {
    "names": names,
    "groups": groups,
    "area": area,
    "harvest": harvest,
    "price": price,
    "yield": yield_t_per_ha,
    "revenue_coeff": revenue_coeff,
    "base_crop_revenue": base_crop_revenue,
    "base_total_area": base_total_area,
    "group_indices": group_indices,
    "base_group_area": base_group_area,
    "base_group_harvest": base_group_harvest,
    "base_group_revenue": base_group_revenue,
    "base_total_revenue": sum(base_crop_revenue),
  }


def read_scenarios(path: Path) -> list[Scenario]:
  """Читает три коэффициента сценария; поддерживает CSV с ',' и ';'."""
  scenarios: list[Scenario] = []
  text = path.read_text(encoding="utf-8-sig")
  first_line = next((line for line in text.splitlines() if line.strip()), "")
  delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
  reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
  expected = ["alpha", "beta", "gamma"]
  if reader.fieldnames != expected:
    raise ValueError(
      "CSV должен иметь ровно три столбца в указанном порядке: "
      "alpha,beta,gamma. Разделитель может быть запятой или точкой с запятой. "
      "Получено: " + repr(reader.fieldnames)
    )

  for row_number, row in enumerate(reader, start=2):
    try:
      values = [row[key].strip() for key in expected]
      # Для CSV с ';' допускается десятичная запятая, обычная для русской локали Excel.
      if delimiter == ";":
        values = [value.replace(",", ".") for value in values]
      alpha, beta, gamma = (float(value) for value in values)
    except (TypeError, ValueError, AttributeError) as exc:
      raise ValueError(
        f"Строка CSV {row_number}: alpha, beta и gamma должны быть числами."
      ) from exc

    if not all(math.isfinite(v) for v in (alpha, beta, gamma)):
      raise ValueError(f"Строка CSV {row_number}: найдены нечисловые значения.")
    if not 0 <= alpha < 1:
      raise ValueError(f"Строка CSV {row_number}: alpha должна быть в [0, 1).")
    if not 0 <= beta < 1:
      raise ValueError(f"Строка CSV {row_number}: beta должна быть в [0, 1).")
    if not 0 < gamma <= 1:
      raise ValueError(f"Строка CSV {row_number}: gamma должна быть в (0, 1].")
    scenarios.append(Scenario(len(scenarios) + 1, alpha, beta, gamma))

  if not scenarios:
    raise ValueError("CSV не содержит сценариев.")
  return scenarios


def solve_scenario(
  data: dict[str, Any],
  scenario: Scenario,
  require_group_gain: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
  """Решает LP для одного набора alpha, beta, gamma методом двойственного симплекса."""
  n = len(CROPS)
  area = data["area"]
  yield_values = data["yield"]
  revenue_coeff = data["revenue_coeff"]
  group_indices = data["group_indices"]
  alpha, beta, gamma = scenario.alpha, scenario.beta, scenario.gamma

  a_ub: list[list[float]] = []
  b_ub: list[float] = []

  def add_ub(coefficients: list[float], rhs: float) -> None:
    a_ub.append(coefficients)
    b_ub.append(rhs)

  # Индивидуальные границы культур: (1-alpha)*a_i <= x_i <= (1+alpha)*a_i.
  # Аналог ограничений M10:N24 в Excel.
  for i, baseline_area in enumerate(area):
    unit = [0.0] * n
    unit[i] = 1.0
    add_ub(unit, (1 + alpha) * baseline_area)
    add_ub([-value for value in unit], -(1 - alpha) * baseline_area)

  # Внешнее перераспределение: (1-beta)*A_g <= sum(x_i по группе) <= (1+beta)*A_g.
  # Это верхняя и нижняя границы площади каждого набора в сводных строках Excel.
  for group in GROUPS:
    indices = group_indices[group]
    row = [1.0 if i in indices else 0.0 for i in range(n)]
    add_ub(row, (1 + beta) * data["base_group_area"][group])
    add_ub([-value for value in row], -(1 - beta) * data["base_group_area"][group])

  # Минимальный валовый сбор: sum(q_i*x_i) >= gamma*H_g^0.
  # Передано linprog как -sum(q_i*x_i) <= -gamma*H_g^0.
  for group in GROUPS:
    indices = group_indices[group]
    row = [(-yield_values[i] if i in indices else 0.0) for i in range(n)]
    add_ub(row, -gamma * data["base_group_harvest"][group])

  # Условие безубыточности каждого набора: R_g(x) >= R_g^0.
  # Флаг --allow-group-losses отключает только эти три ограничения для сравнения моделей.
  if require_group_gain:
    for group in GROUPS:
      indices = group_indices[group]
      row = [(-revenue_coeff[i] if i in indices else 0.0) for i in range(n)]
      add_ub(row, -data["base_group_revenue"][group])

  # Сумма посевных площадей сохраняется: sum(x_i) = sum(a_i).
  a_eq = [[1.0] * n]
  b_eq = [data["base_total_area"]]

  # linprog минимизирует c^T x, поэтому для максимизации выручки используем -c.
  # Флаг --non-negative явно задаёт x_i >= 0. Без флага переменные свободны по знаку,
  # но индивидуальные нижние границы всё равно вынуждают x_i > 0 при alpha < 1.
  bounds = [(0.0, None) for _ in range(n)]
  result = linprog(
    c=[-value for value in revenue_coeff],
    A_ub=a_ub,
    b_ub=b_ub,
    A_eq=a_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs-ds",
  )

  base_total_revenue = data["base_total_revenue"]
  if not result.success:
    summary = {
      "scenario": scenario.number,
      "alpha": alpha,
      "beta": beta,
      "gamma": gamma,
      "solver_status": result.status,
      "solver_message": result.message,
      "baseline_revenue_billion_rub": base_total_revenue,
      "optimized_revenue_billion_rub": None,
      "revenue_gain_billion_rub": None,
      "total_area_thousand_ha": None,
    }
    return summary, [], []

  x_values = [float(value) for value in result.x]
  crop_revenue = [revenue_coeff[i] * x_values[i] for i in range(n)]
  crop_harvest = [yield_values[i] * x_values[i] for i in range(n)]
  optimized_total_revenue = sum(crop_revenue)

  group_rows: list[dict[str, Any]] = []
  group_gains: dict[str, float] = {}
  for group in GROUPS:
    indices = group_indices[group]
    optimized_area = sum(x_values[i] for i in indices)
    optimized_harvest = sum(crop_harvest[i] for i in indices)
    optimized_revenue = sum(crop_revenue[i] for i in indices)
    gain = optimized_revenue - data["base_group_revenue"][group]
    if abs(gain) < 1e-7:
      gain = 0.0  # убираем только численные остатки около нуля
    group_gains[group] = gain
    group_rows.append({
      "scenario": scenario.number,
      "alpha": alpha,
      "beta": beta,
      "gamma": gamma,
      "group": group,
      "baseline_area_thousand_ha": data["base_group_area"][group],
      "optimized_area_thousand_ha": optimized_area,
      "area_change_thousand_ha": optimized_area - data["base_group_area"][group],
      "baseline_harvest_thousand_t": data["base_group_harvest"][group],
      "optimized_harvest_thousand_t": optimized_harvest,
      "harvest_minimum_thousand_t": gamma * data["base_group_harvest"][group],
      "baseline_revenue_billion_rub": data["base_group_revenue"][group],
      "optimized_revenue_billion_rub": optimized_revenue,
      "revenue_gain_billion_rub": gain,
    })

  crop_rows = []
  for i, crop in enumerate(CROPS):
    crop_rows.append({
      "scenario": scenario.number,
      "alpha": alpha,
      "beta": beta,
      "gamma": gamma,
      "group": crop.group,
      "crop": crop.name,
      "baseline_area_thousand_ha": area[i],
      "optimized_area_thousand_ha": x_values[i],
      "area_change_thousand_ha": x_values[i] - area[i],
      "model_yield_t_per_ha": yield_values[i],
      "optimized_harvest_thousand_t": crop_harvest[i],
      "price_rub_per_t": data["price"][i],
      "baseline_revenue_billion_rub": data["base_crop_revenue"][i],
      "optimized_revenue_billion_rub": crop_revenue[i],
      "revenue_gain_billion_rub": crop_revenue[i] - data["base_crop_revenue"][i],
    })

  summary = {
    "scenario": scenario.number,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "solver_status": result.status,
    "solver_message": result.message,
    "baseline_revenue_billion_rub": base_total_revenue,
    "optimized_revenue_billion_rub": optimized_total_revenue,
    "revenue_gain_billion_rub": optimized_total_revenue - base_total_revenue,
    "winter_cereals_gain_billion_rub": group_gains[GROUPS[0]],
    "oilseeds_gain_billion_rub": group_gains[GROUPS[1]],
    "forage_gain_billion_rub": group_gains[GROUPS[2]],
    "total_area_thousand_ha": sum(x_values),
  }
  return summary, group_rows, crop_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  """Записывает список однотипных словарей в CSV с UTF-8 BOM для Excel."""
  if not rows:
    # Создаём пустой файл с корректным UTF-8 BOM, когда ни один сценарий не решён.
    path.write_text("\ufeff", encoding="utf-8")
    return
  fieldnames = list(rows[0].keys())
  with path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def process(
  scenarios_path: Path,
  output_dir: Path,
  allow_group_losses: bool,
) -> int:
  """Основной процесс решения задачи."""
  scenarios = read_scenarios(scenarios_path)

  data = build_model_data()
  all_summaries: list[dict[str, Any]] = []
  all_groups: list[dict[str, Any]] = []
  all_crops: list[dict[str, Any]] = []

  for scenario in scenarios:
    summary, group_rows, crop_rows = solve_scenario(
      data=data,
      scenario=scenario,
      require_group_gain=not allow_group_losses,
    )
    all_summaries.append(summary)
    all_groups.extend(group_rows)
    all_crops.extend(crop_rows)

  output_dir.mkdir(parents=True, exist_ok=True)
  summaries_path = output_dir / "summaries.csv"
  groups_path = output_dir / "groups.csv"
  crops_path = output_dir / "crops.csv"
  write_csv(summaries_path, all_summaries)
  write_csv(groups_path, all_groups)
  write_csv(crops_path, all_crops)

  successful = sum(row["solver_status"] == 0 for row in all_summaries)
  print(f"Сценариев обработано: {len(scenarios)}; успешно решено: {successful}.")
  print(f"Базовая выручка: {data['base_total_revenue']:.6f} млрд руб.")
  print(
    f"Условие выручки каждой группы: "
    f"{'отключено' if allow_group_losses else 'включено'}."
  )
  print(f"Сводка: {summaries_path}")
  print(f"По группам: {groups_path}")
  print(f"По культурам: {crops_path}")

  failed = [row for row in all_summaries if row["solver_status"] != 0]
  for row in failed:
    print(
      f"Сценарий {row['scenario']} (alpha={row['alpha']}, beta={row['beta']}, "
      f"gamma={row['gamma']}): {row['solver_message']}",
      file=sys.stderr,
    )
  return 0 if successful == len(scenarios) else 1


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
  """Разобрать аргументы командной строки."""
  parser = argparse.ArgumentParser(
    description="Решение модели структуры посевных площадей методом двойственного симплекса."
  )
  parser.add_argument(
    "scenarios",
    type=Path,
    help="CSV с фиксированными столбцами alpha, beta, gamma (одна строка на сценарий)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs"),
    help="Каталог для результатов (по умолчанию: outputs)",
  )
  parser.add_argument(
    "--allow-group-losses",
    action="store_true",
    help="Отключить ограничения R_g >= R_g^0 и разрешить группе терять выручку",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  """Запустить процесс и обработать ожидаемые ошибки."""
  args = parse_arguments(argv)
  try:
    return process(
      args.scenarios,
      args.output_dir,
      args.allow_group_losses,
    )
  except (OSError, ValueError) as exc:
    print(f"Ошибка входного CSV: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
