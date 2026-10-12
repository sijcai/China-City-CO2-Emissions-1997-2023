# Random Forest modelling of city-year emissions across seven sectors
# This script documents hyperparameter tuning, cross-validation, and model evaluation using low-anomaly samples.

library(dplyr)
library(mlr3verse)
library(paradox)
library(ranger)

# Required input files:
# {sector}_train_lof.csv and {sector}_test_lof.csv
# {sector} refers to one of the following sector identifiers:
# petroleum_coal_other_fuels, chemical_products, building_materials, iron_and_steel, nonferrous_metals, electricity_and_heat, other.
# Example:
# petroleum_coal_other_fuels_train_lof.csv
# petroleum_coal_other_fuels_test_lof.csv

# Required columns:
# City_id, Year, Emission_Mt, label_lof, and the sector-specific predictors below.

# -----------------------------
# 1. Define sector-specific predictors and selected parameters
# -----------------------------
sector_settings <- list(
  petroleum_coal_other_fuels = list(
    predictors = c(
      "Average_employees_person",
      "Raw.Coal_10000.tons",
      "Coke_10000.tons",
      "Administrative_Land_Area_squarekilometer",
      "Crude.Oil.Processing_10000.tons",
      "CLCD_corpland_m2",
      "CLCD_impervious_m2",
      "Coal_plant_capacity_mw"
    ),
    num.trees = 540L,
    min.node.size = 3L,
    mtry = 5L
  ),

  chemical_products = list(
    predictors = c(
      "Average_employees_person",
      "Synthetic.Ammonia_10000.tons",
      "Ethylene_10000.tons",
      "Coal_plant_capacity_mw",
      "Manufactured.Gas.Total.Gas.Supplied..10.000m3.",
      "CLCD_impervious_m2",
      "CLCD_water_m2",
      "Calcium.Carbide_10000.tons"
    ),
    num.trees = 180L,
    min.node.size = 2L,
    mtry = 3L
  ),

  building_materials = list(
    predictors = c(
      "Cement_10000.tons",
      "CLCD_forest_m2",
      "Local_General_Public_Budget_Expenditure_tenthousandyuan",
      "Total_industrial_output_value_thousand_yuan"
    ),
    num.trees = 800L,
    min.node.size = 2L,
    mtry = 2L
  ),

  iron_and_steel = list(
    predictors = c(
      "Crude.Steel_10000.tons",
      "Pig.Iron_10000.tons",
      "Average_employees_person",
      "Tem_avg_centigrade"
    ),
    num.trees = 820L,
    min.node.size = 3L,
    mtry = 3L
  ),

  nonferrous_metals = list(
    predictors = c(
      "Primary.Aluminum_10000.tons",
      "Total_industrial_output_value_thousand_yuan",
      "CLCD_impervious_m2",
      "CLCD_water_m2",
      "Average_employees_person",
      "Annual.sewage.discharge..ten.thousand.cubic.meters.",
      "CLCD_corpland_m2"
    ),
    num.trees = 300L,
    min.node.size = 3L,
    mtry = 3L
  ),

  electricity_and_heat = list(
    predictors = c(
      "Coal_plant_capacity_mw",
      "Total_industrial_output_value_thousand_yuan",
      "Comprehensive.production.capacity..10.000.cubic.meters.day.",
      "Total.water.supply..10.000.cubic.meters."
    ),
    num.trees = 940L,
    min.node.size = 3L,
    mtry = 3L
  ),

  other = list(
    predictors = c(
      "Built_up_area_squarekilometer_shixiaqv",
      "Public_green_areas_hectare_shixiaqv",
      "Length.of.water.supply.pipelines..kilometers.",
      "Local_General_Public_Budget_Expenditure_tenthousandyuan",
      "Annual.sewage.discharge..ten.thousand.cubic.meters."
    ),
    num.trees = 360L,
    min.node.size = 2L,
    mtry = 3L
  )
)

dir.create("results", showWarnings = FALSE)

for (sector in names(sector_settings)) {

  settings <- sector_settings[[sector]]
  predictors <- settings$predictors

  # -----------------------------
  # 2. Select low-anomaly training and testing samples
  # -----------------------------
  train_data <- read.csv(
    paste0(sector, "_train_lof.csv")
  ) %>%
    filter(label_lof == "low_anomaly")

  test_data <- read.csv(
    paste0(sector, "_test_lof.csv")
  ) %>%
    filter(label_lof == "low_anomaly")

  train_model_data <- train_data %>%
    select(all_of(c(predictors, "Emission_Mt")))

  test_model_data <- test_data %>%
    select(all_of(c(predictors, "Emission_Mt")))

  task_train <- as_task_regr(
    train_model_data,
    target = "Emission_Mt"
  )

  task_test <- as_task_regr(
    test_model_data,
    target = "Emission_Mt"
  )

  # -----------------------------
  # 3. Tune hyperparameters using five-fold cross-validation
  # -----------------------------
  set.seed(42)

  search_space <- ps(
    num.trees = p_int(
      lower = 1,
      upper = 50,
      trafo = function(x) 20L * x
    ),
    min.node.size = p_int(
      lower = 2,
      upper = 7
    ),
    mtry = p_int(
      lower = 2,
      upper = min(7L, length(predictors))
    )
  )

  tuner <- auto_tuner(
    learner = lrn(
      "regr.ranger",
      importance = "permutation",
      num.threads = 1L
    ),
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.rsq"),
    search_space = search_space,
    tuner = tnr("random_search"),
    term_evals = 10
  )

  tuner$train(task_train)

  saveRDS(
    tuner$tuning_result,
    paste0("results/", sector, "_tuning_result.rds")
  )

  # -----------------------------
  # 4. Define the model using the selected research parameters
  # -----------------------------
  # The original scripts use fixed selected parameters after tuning.
  # These values are retained above for the final research models.
  learner_rf <- lrn(
    "regr.ranger",
    importance = "permutation",
    num.trees = settings$num.trees,
    min.node.size = settings$min.node.size,
    mtry = settings$mtry,
    num.threads = 1L
  )

  # -----------------------------
  # 5. Evaluate using ten-fold cross-validation
  # -----------------------------
  set.seed(42)

  cv_result <- resample(
    task_train,
    learner_rf$clone(deep = TRUE),
    resampling = rsmp("cv", folds = 10)
  )

  cv_scores <- cv_result$score(
    list(
      msr("regr.rsq"),
      msr("regr.rmse")
    )
  )

  write.csv(
    as.data.frame(cv_scores),
    paste0("results/", sector, "_cv_scores.csv"),
    row.names = FALSE
  )

  # -----------------------------
  # 6. Train the final model and predict testing samples
  # -----------------------------
  set.seed(42)

  learner_rf$train(task_train)

  prediction <- learner_rf$predict(task_test)

  observed <- prediction$truth
  predicted <- prediction$response

  # -----------------------------
  # 7. Calculate testing performance
  # -----------------------------
  r2 <- unname(prediction$score(msr("regr.rsq")))
  rmse <- unname(prediction$score(msr("regr.rmse")))
  mae <- unname(prediction$score(msr("regr.mae")))

  # Following the source scripts, percentage-error metrics use
  # observations greater than 1 Mt to reduce small-denominator effects.
  valid <- is.finite(observed) &
    is.finite(predicted) &
    observed > 1

  percentage_error <- (
    predicted[valid] - observed[valid]
  ) / observed[valid]

  performance <- data.frame(
    Sector = sector,
    R2 = r2,
    RMSE_Mt = rmse,
    MAE_Mt = mae,
    MPE_percent = if (any(valid)) {
      mean(percentage_error) * 100
    } else {
      NA_real_
    },
    MAPE_percent = if (any(valid)) {
      mean(abs(percentage_error)) * 100
    } else {
      NA_real_
    },
    n_test = length(observed),
    n_percentage = sum(valid)
  )

  write.csv(
    performance,
    paste0("results/", sector, "_test_performance.csv"),
    row.names = FALSE
  )

  # -----------------------------
  # 8. Save predictions, variable importance, and fitted models
  # -----------------------------
  test_predictions <- test_data %>%
    select(City_id, Year) %>%
    mutate(
      Observed_emission_Mt = observed,
      Predicted_emission_Mt = predicted
    )

  write.csv(
    test_predictions,
    paste0("results/", sector, "_test_predictions.csv"),
    row.names = FALSE
  )

  importance_values <- learner_rf$importance()

  variable_importance <- data.frame(
    Predictor = names(importance_values),
    Importance = unname(importance_values)
  )

  write.csv(
    variable_importance,
    paste0("results/", sector, "_variable_importance.csv"),
    row.names = FALSE
  )

  saveRDS(
    learner_rf,
    paste0("results/", sector, "_rf_model.rds")
  )
}
