# Линейная оптимизация структуры посевных площадей за 2020 год.
# Модель повторяет таблицу Excel «Модель LP» и добавляет условия:
# выручка каждого из трёх наборов после перераспределения >= базовой выручки.
#
# Зависимость от пакета: lpSolve
# Установка один раз: install.packages("lpSolve")
# Запуск из каталога файла: Rscript model_oilseeds_winter_forage_2020.R
#
# Единицы:
#   a, x — тыс. га;
#   q — т/га;
#   p — руб./т;
#   q*x — тыс. т;
#   p*q*x — тыс. руб.;
#   (p*q*x)/1e6 — млрд руб.
# Масштабирование цели с тыс. рублей на млрд рублей не меняет оптимальный план.

if (!requireNamespace("lpSolve", quietly = TRUE)) {
    stop("Не найден пакет lpSolve. Установите его командой install.packages('lpSolve').")
}

# Данные перенесены из книги Excel. Для сопоставимости с ней используем ту же
# модельную урожайность q = валовой сбор 2020 / посевная площадь 2020.
# В Excel это столбец G; цена p — столбец F; базовая площадь a — столбец D.
# Если позже будут уточнены коэффициенты, достаточно изменить значения здесь.
crops <- data.frame(
    group = c(
        rep("Озимые зерновые", 4),
        rep("Масличные культуры", 6),
        rep("Кормовые культуры", 5)
    ),
    crop = c(
        "Пшеница озимая", "Рожь озимая", "Ячмень озимый", "Тритикале",
        "Подсолнечник", "Лён-кудряш", "Соя", "Горчица", "Рапс",
        "Прочие масличные культуры (остаток)",
        "Кормовые корнеплоды", "Кукуруза на корм", "Сено многолетних трав",
        "Сено однолетних трав", "Прочие кормовые культуры (остаток)"
    ),
    area_2020_thousand_ha = c(
        16914, 980, 731, 98,
        8545, 1030, 2858, 201, 1488, 276,
        16, 1258, 9927, 3373, 177
    ),
    harvest_2020_thousand_t = c(
        63200, 2400, 2300, 300,
        13300, 788, 4308, 103, 2572, 129,
        400, 24800, 8300, 2100, 1770
    ),
    price_rub_per_t = c(
        12240, 9662, 9742, 11900,
        24540, 30000, 26859, 29000, 27570, 30000,
        3000, 1500, 7500, 7500, 3000
    ),
    stringsAsFactors = FALSE
)

# q_i = E_i / D_i — аналог модельной урожайности G10:G24 в Excel.
# Это отношение валового сбора к посевной площади, а не официальная урожайность
# с убранной площади. Так базовый модельный сбор точно совпадает с исходным.
crops$yield_t_per_ha <- crops$harvest_2020_thousand_t / crops$area_2020_thousand_ha

# r_i = p_i*q_i — аналог I10:I24, выручка с одного гектара в руб./га.
crops$revenue_per_ha_rub <- crops$price_rub_per_t * crops$yield_t_per_ha

# Коэффициенты цели в млрд руб. x_i измеряется в тыс. га:
# c_i*x_i = p_i*q_i*x_i / 1 000 000 — аналог суммирования O10:O24,
# но в более читаемой единице измерения (млрд руб., а не тыс. руб.).
crops$revenue_coeff_billion_rub <- crops$revenue_per_ha_rub / 1e6

# Базовые значения, соответствующие x_i = a_i в Excel.
crops$base_model_harvest_thousand_t <- crops$yield_t_per_ha * crops$area_2020_thousand_ha
crops$base_revenue_billion_rub <- crops$revenue_coeff_billion_rub * crops$area_2020_thousand_ha

# Границы наборов: в Excel — сводные строки 29:31.
group_names <- c("Озимые зерновые", "Масличные культуры", "Кормовые культуры")
group_id <- match(crops$group, group_names)
group_rows <- lapply(seq_along(group_names), function(g) which(group_id == g))
names(group_rows) <- group_names

base_group_area <- vapply(group_rows, function(ix) sum(crops$area_2020_thousand_ha[ix]), numeric(1))
base_group_harvest <- vapply(group_rows, function(ix) sum(crops$base_model_harvest_thousand_t[ix]), numeric(1))
base_group_revenue <- vapply(group_rows, function(ix) sum(crops$base_revenue_billion_rub[ix]), numeric(1))
base_total_area <- sum(crops$area_2020_thousand_ha)
base_total_revenue <- sum(crops$base_revenue_billion_rub)

# Excel-параметры сценария:
# B6 — α (внутреннее отклонение), D6 — β (внешнее отклонение), F6 — γ (минимум сбора).
# Все комбинации: 3 значения α * 3 значения β * 3 значения γ = 27 сценариев.
scenarios <- expand.grid(
    gamma = c(0.90, 0.95, 0.98),
    beta = c(0.05, 0.075, 0.10),
    alpha = c(0.10, 0.15, 0.20),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
)
scenarios <- scenarios[order(scenarios$alpha, scenarios$beta, scenarios$gamma), ]
rownames(scenarios) <- NULL

# Функция собирает матрицу линейных ограничений для заданного сценария.
solve_scenario <- function(alpha, beta, gamma, scenario_number) {
    n <- nrow(crops)
    constraint_rows <- list()
    constraint_dirs <- character()
    constraint_rhs <- numeric()
    constraint_names <- character()

    add_constraint <- function(coef, direction, rhs, name) {
        constraint_rows[[length(constraint_rows) + 1L]] <<- as.numeric(coef)
        constraint_dirs <<- c(constraint_dirs, direction)
        constraint_rhs <<- c(constraint_rhs, as.numeric(rhs))
        constraint_names <<- c(constraint_names, name)
    }

    a <- crops$area_2020_thousand_ha
    q <- crops$yield_t_per_ha
    revenue_coeff <- crops$revenue_coeff_billion_rub

    # Индивидуальные пределы культур (Excel M10:N24):
    # (1 - α)*a_i <= x_i <= (1 + α)*a_i.
    # Для lpSolve эти неравенства добавляются двумя строками на каждую культуру.
    for (i in seq_len(n)) {
        unit <- numeric(n)
        unit[i] <- 1
        add_constraint(unit, ">=", (1 - alpha) * a[i], paste0("crop_lower_", i))
        add_constraint(unit, "<=", (1 + alpha) * a[i], paste0("crop_upper_", i))
    }

    # Внешнее перераспределение между наборами (Excel — сводные строки 29:31):
    # (1 - β)*A_g <= sum(x_i in g) <= (1 + β)*A_g.
    for (g in seq_along(group_rows)) {
        ix <- group_rows[[g]]
        group_coef <- numeric(n)
        group_coef[ix] <- 1
        add_constraint(group_coef, ">=", (1 - beta) * base_group_area[g], paste0("group_area_lower_", g))
        add_constraint(group_coef, "<=", (1 + beta) * base_group_area[g], paste0("group_area_upper_", g))
    }

    # Общая площадь неизменна: sum(x_i) = sum(a_i), аналог общей строки Excel.
    add_constraint(rep(1, n), "=", base_total_area, "total_area_fixed")

    # Минимальный валовой сбор каждого набора (аналог столбцов G/H сводной таблицы):
    # sum(q_i*x_i in g) >= γ*H_g^0, где H_g^0 = sum(q_i*a_i).
    for (g in seq_along(group_rows)) {
        ix <- group_rows[[g]]
        harvest_coef <- numeric(n)
        harvest_coef[ix] <- q[ix]
        add_constraint(harvest_coef, ">=", gamma * base_group_harvest[g], paste0("harvest_min_", g))
    }

    # НОВОЕ УСЛОВИЕ для каждой группы (прирост из выручки, которую вы добавили в Excel):
    # ΔR_g = sum(p_i*q_i*x_i) - R_g^0 >= 0.
    # Коэффициенты и правая часть представлены в млрд руб., чтобы числа были меньше.
    # В Excel текущая выручка культуры = I_i*J_i (столбец O), базовая — I_i*D_i.
    for (g in seq_along(group_rows)) {
        ix <- group_rows[[g]]
        revenue_constraint <- numeric(n)
        revenue_constraint[ix] <- revenue_coeff[ix]
        add_constraint(
            revenue_constraint,
            ">=",
            base_group_revenue[g],
            paste0("group_revenue_not_below_base_", g)
        )
    }

    constraint_matrix <- do.call(rbind, constraint_rows)
    rownames(constraint_matrix) <- constraint_names
    colnames(constraint_matrix) <- crops$crop

    # Целевая функция: max sum(p_i*q_i*x_i); здесь результат в млрд руб.
    fit <- lpSolve::lp(
        direction = "max",
        objective.in = revenue_coeff,
        const.mat = constraint_matrix,
        const.dir = constraint_dirs,
        const.rhs = constraint_rhs,
        all.int = FALSE,
        all.bin = FALSE
    )

    if (fit$status != 0) {
        return(list(
            status = fit$status,
            scenario = data.frame(
                scenario = scenario_number, alpha = alpha, beta = beta, gamma = gamma,
                solver_status = fit$status, total_area_thousand_ha = NA_real_,
                baseline_revenue_billion_rub = base_total_revenue,
                optimized_revenue_billion_rub = NA_real_,
                revenue_gain_billion_rub = NA_real_,
                stringsAsFactors = FALSE
            ),
            groups = data.frame(),
            allocations = data.frame()
        ))
    }

    x <- fit$solution
    crop_revenue <- revenue_coeff * x
    crop_harvest <- q * x
    total_revenue <- sum(crop_revenue)
    group_out <- lapply(seq_along(group_rows), function(g) {
        ix <- group_rows[[g]]
        area_now <- sum(x[ix])
        harvest_now <- sum(crop_harvest[ix])
        revenue_now <- sum(crop_revenue[ix])
        data.frame(
            scenario = scenario_number,
            alpha = alpha, beta = beta, gamma = gamma,
            group = group_names[g],
            baseline_area_thousand_ha = base_group_area[g],
            optimized_area_thousand_ha = area_now,
            area_change_thousand_ha = area_now - base_group_area[g],
            baseline_harvest_thousand_t = base_group_harvest[g],
            optimized_harvest_thousand_t = harvest_now,
            harvest_minimum_thousand_t = gamma * base_group_harvest[g],
            baseline_revenue_billion_rub = base_group_revenue[g],
            optimized_revenue_billion_rub = revenue_now,
            revenue_gain_billion_rub = revenue_now - base_group_revenue[g],
            stringsAsFactors = FALSE
        )
    })
    groups <- do.call(rbind, group_out)

    # Из-за численной погрешности LP-решателя значения порядка 1e-8 млрд руб.
    # могут отображаться как крошечные минусы; для отчёта нормализуем только такие числа.
    tiny <- 1e-7
    groups$revenue_gain_billion_rub[abs(groups$revenue_gain_billion_rub) < tiny] <- 0

    scenario_result <- data.frame(
        scenario = scenario_number,
        alpha = alpha, beta = beta, gamma = gamma,
        solver_status = fit$status,
        total_area_thousand_ha = sum(x),
        baseline_revenue_billion_rub = base_total_revenue,
        optimized_revenue_billion_rub = total_revenue,
        revenue_gain_billion_rub = total_revenue - base_total_revenue,
        winter_cereals_gain_billion_rub = groups$revenue_gain_billion_rub[groups$group == "Озимые зерновые"],
        oilseeds_gain_billion_rub = groups$revenue_gain_billion_rub[groups$group == "Масличные культуры"],
        forage_gain_billion_rub = groups$revenue_gain_billion_rub[groups$group == "Кормовые культуры"],
        stringsAsFactors = FALSE
    )

    allocations <- data.frame(
        scenario = scenario_number,
        alpha = alpha, beta = beta, gamma = gamma,
        group = crops$group,
        crop = crops$crop,
        baseline_area_thousand_ha = a,
        optimized_area_thousand_ha = x,
        area_change_thousand_ha = x - a,
        model_yield_t_per_ha = q,
        optimized_harvest_thousand_t = crop_harvest,
        price_rub_per_t = crops$price_rub_per_t,
        baseline_revenue_billion_rub = crops$base_revenue_billion_rub,
        optimized_revenue_billion_rub = crop_revenue,
        revenue_gain_billion_rub = crop_revenue - crops$base_revenue_billion_rub,
        stringsAsFactors = FALSE
    )

    list(status = fit$status, scenario = scenario_result, groups = groups, allocations = allocations)
}

# Запускаем 27 сценариев и сохраняем три CSV:
# 1) scenario_results.csv — итог по каждому набору параметров;
# 2) group_results.csv — площадь, сбор и прирост выручки каждого набора;
# 3) crop_allocations.csv — оптимальная площадь и выручка каждой культуры.
scenario_results <- vector("list", nrow(scenarios))
group_results <- vector("list", nrow(scenarios))
allocation_results <- vector("list", nrow(scenarios))

for (s in seq_len(nrow(scenarios))) {
    sc <- scenarios[s, ]
    result <- solve_scenario(sc$alpha, sc$beta, sc$gamma, s)
    scenario_results[[s]] <- result$scenario
    group_results[[s]] <- result$groups
    allocation_results[[s]] <- result$allocations
}

scenario_results <- do.call(rbind, scenario_results)
group_results <- do.call(rbind, group_results)
allocation_results <- do.call(rbind, allocation_results)

write.csv(scenario_results, "scenario_results.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(group_results, "group_results.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(allocation_results, "crop_allocations.csv", row.names = FALSE, fileEncoding = "UTF-8")

# Выводим ключевые результаты в консоль; деньги в этом отчёте — млрд рублей.
cat("Всего сценариев:", nrow(scenario_results), "\n")
cat("Успешно решено:", sum(scenario_results$solver_status == 0), "\n")
cat("Базовая выручка:", round(base_total_revenue, 3), "млрд руб.\n")
cat("\nПервые 27 результатов сохранены в scenario_results.csv\n")
print(scenario_results, row.names = FALSE, digits = 6)

# Самопроверка: каждое успешно найденное решение должно удовлетворять условиям
# непросадки выручки по всем трём наборам (с допуском численной погрешности).
if (any(scenario_results$solver_status == 0)) {
    successful <- subset(scenario_results, solver_status == 0)
    gains <- successful[, c(
        "winter_cereals_gain_billion_rub",
        "oilseeds_gain_billion_rub",
        "forage_gain_billion_rub"
    )]
    if (any(as.matrix(gains) < -1e-6)) {
        warning("Есть прирост выручки группы ниже нуля; проверьте допуски и ограничения.")
    }
}
