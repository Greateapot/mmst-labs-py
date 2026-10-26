library(lpSolve)

# Исходные данные за 2020 год
crop <- c(
  "Подсолнечник",
  "Лён-кудряш",
  "Соя",
  "Горчица",
  "Рапс"
)

area <- c(8545, 1030, 2858, 201, 1488)
harvest <- c(13300, 788, 4308, 103, 2572)

# Цены производителей, руб./т
# Для льна и горчицы указаны условные сценарные значения
price <- c(24540, 32650, 26859, 28552, 27570)

# Валовой сбор на гектар, т/га
yield_per_ha <- harvest / area

# Выручка на гектар, руб./га
revenue_per_ha <- yield_per_ha * price

# Общая площадь пяти культур, тыс. га
total_area <- sum(area)

# Решение задачи при заданном допустимом отклонении
solve_model <- function(tolerance) {
  lower <- area * (1 - tolerance)
  upper <- area * (1 + tolerance)

  # Первая строка: общая площадь.
  # Следующие строки: верхние и нижние границы культур.
  constraints <- rbind(
    rep(1, length(crop)),
    diag(length(crop)),
    -diag(length(crop))
  )

  directions <- c("=", rep("<=", 2 * length(crop)))
  limits <- c(total_area, upper, -lower)

  result <- lp(
    direction = "max",
    objective.in = revenue_per_ha,
    const.mat = constraints,
    const.dir = directions,
    const.rhs = limits
  )

  if (result$status != 0) {
    stop(
      "Оптимальное решение не найдено. Код статуса: ",
      result$status
    )
  }

  solution <- result$solution

  # Сравнение с исходной структурой 2020 года
  initial_revenue <- sum(area * revenue_per_ha)
  optimal_revenue <- result$objval

  table <- data.frame(
    `Культура` = crop,
    `Площадь_2020_тыс_га` = area,
    `Оптимальная_площадь_тыс_га` = solution,
    `Изменение_тыс_га` = solution - area,
    `Минимум_тыс_га` = lower,
    `Максимум_тыс_га` = upper,
    check.names = FALSE
  )

  list(
    tolerance = tolerance,
    table = table,
    initial_revenue_billion = initial_revenue,
    optimal_revenue_billion = optimal_revenue,
    gain_billion = (optimal_revenue - initial_revenue),
    status = result$status
  )
}

# Решения для трёх значений допустимого отклонения
results <- lapply(
  c(0.20, 0.15, 0.02),
  solve_model
)

for (result in results) {
  cat(
    "\nДопустимое отклонение:",
    result$tolerance * 100, "%\n"
  )

  print(result$table, row.names = FALSE, digits = 8)

  cat(
    "Выручка до оптимизации, тыс руб.:",
    round(result$initial_revenue_billion, 3), "\n"
  )

  cat(
    "Выручка после оптимизации, тыс руб.:",
    round(result$optimal_revenue_billion, 3), "\n"
  )

  cat(
    "Прирост выручки, тыс руб.:",
    round(result$gain_billion, 3), "\n"
  )
}
