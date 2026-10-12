# Random Forest tuning and prediction intervals across seven sectors
# This script selects hyperparameters by cross-validation and screens potentially high-uncertainty samples using 95% prediction intervals.

library(dplyr)
library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(paradox)
library(ranger)
library(RFpredInterval)

# Required input files:
# {sector}_train_lof.csv and {sector}_test_lof.csv
# {sector} refers to:
# petroleum_coal_other_fuels, chemical_products, building_materials,iron_and_steel, nonferrous_metals, electricity_and_heat, other.

# Required columns:
# City_id, Year, Emission_Mt, label_lof, and the sector-specific predictors below.

# -----------------------------
# 1. Define sector-specific predictors
# -----------------------------
sector_predictors <- list(
  petroleum_coal_other_fuels = c(
    "Emp-PCFPI",
    "RCoal",
    "Coke",
    "CrudeOil",
    "ImpArea",
    "PopDensity",
    "CoalCap"
  ),

  chemical_products = c(
    "Emp-Chem",
    "SynAmmonia",
    "CPlantCap",
    "Ethylene",
    "WaterArea",
    "ImpArea",
    "AvgHumidity",
    "MG_Supply"
  ),

  building_materials = c(
    "Cement",
    "GreenArea",
    "ForestArea",
    "LGExp"
  ),

  iron_and_steel = c(
    "PIron",
    "CSteel",
    "Emp-ISI",
    "ImpArea",
    "AvgTemp"
  ),

  nonferrous_metals = c(
    "PrimAlum",
    "Emp-NFMSI",
    "Output-NFMSI",
    "GrassArea",
    "CropArea",
    "ImpArea",
    "SewageDis"
  ),

  electricity_and_heat = c(
    "CoalCap",
    "Output-EHS",
    "ImpArea",
    "ProdCap",
    "WaterSupply",
    "WaterArea"
  ),

  other = c(
    "BuiltArea",
    "GreenArea",
    "WaterPipeLen",
    "GDP",
    "PopDensity"
  )
)

dir.create("results", showWarnings = FALSE)

all_flagged_records <- list()

for (sector in names(sector_predictors)) {

  predictors <- sector_predictors[[sector]]

  # -----------------------------
  # 2. Read training and testing data
  # -----------------------------
  train_data <- read.csv(
    paste0(sector, "_train_lof.csv"),
    check.names = FALSE
  )

  test_data <- read.csv(
    paste0(sector, "_test_lof.csv"),
    check.names = FALSE
  )

  # -----------------------------
  # 3. Select low-anomaly and high-anomaly samples
  # -----------------------------
  low_anomaly_train <- train_data %>%
    filter(label_lof == "low_anomaly")

  high_anomaly_train <- train_data %>%
    filter(label_lof == "high_anomaly")

  low_anomaly_test <- test_data %>%
    filter(label_lof == "low_anomaly")

  high_anomaly_test <- test_data %>%
    filter(label_lof == "high_anomaly")

  # -----------------------------
  # 4. Prepare the training dataset
  # -----------------------------
  model_columns <- c("Emission_Mt", predictors)

  training_data <- low_anomaly_train %>%
    select(all_of(model_columns))

  # Use internal names for compatibility with formula-based functions.
  # Apply the same mapping to all modelling datasets.
  internal_names <- c(
    "Emission_Mt",
    paste0("X", seq_along(predictors))
  )

  names(training_data) <- internal_names

  task_train <- as_task_regr(
    training_data,
    target = "Emission_Mt"
  )

  # -----------------------------
  # 5. Tune hyperparameters using five-fold cross-validation
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

  best_parameters <-
    tuner$tuning_result$learner_param_vals[[1]]

  saveRDS(
    best_parameters,
    paste0("results/", sector, "_best_parameters.rds")
  )

  # The auto-tuner refits the selected model on the full training task.
  saveRDS(
    tuner,
    paste0("results/", sector, "_rf_model.rds")
  )

  # -----------------------------
  # 6. Evaluate the selected model on low-anomaly testing samples
  # -----------------------------
  if (nrow(low_anomaly_test) > 0) {

    testing_data <- low_anomaly_test %>%
      select(all_of(model_columns))

    names(testing_data) <- internal_names

    task_test <- as_task_regr(
      testing_data,
      target = "Emission_Mt"
    )

    prediction <- tuner$predict(task_test)

    observed <- prediction$truth
    predicted <- prediction$response

    # Percentage-error metrics use observed emissions greater than 1 Mt,
    # following the supplied research script.
    valid <- observed > 1
    percentage_error <- (
      predicted[valid] - observed[valid]
    ) / observed[valid]

    performance <- data.frame(
      Sector = sector,
      R2 = unname(prediction$score(msr("regr.rsq"))),
      RMSE_Mt = unname(prediction$score(msr("regr.rmse"))),
      MAE_Mt = unname(prediction$score(msr("regr.mae"))),
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

    test_predictions <- low_anomaly_test %>%
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
  }

  # -----------------------------
  # 7. Prepare parameters for interval prediction
  # -----------------------------
  # Map cross-validation-selected ranger parameters to rfsrc names.
  interval_parameters <- list(
    ntree = best_parameters$num.trees,
    nodesize = best_parameters$min.node.size,
    mtry = best_parameters$mtry
  )

  high_anomaly_samples <- list(
    train = high_anomaly_train,
    test = high_anomaly_test
  )

  for (split in names(high_anomaly_samples)) {

    screening_data <- high_anomaly_samples[[split]]

    interval_results <- screening_data %>%
      select(
        any_of(c("Province", "City")),
        City_id,
        Year,
        Emission_Mt
      )

    interval_results$Sector <- rep(sector, nrow(screening_data))
    interval_results$Dataset <- rep(split, nrow(screening_data))

    if (nrow(screening_data) == 0) {

      # Preserve output headers when no high-anomaly samples are present.
      interval_results$lower <- numeric(0)
      interval_results$upper <- numeric(0)
      interval_results$Potential_high_uncertainty <- logical(0)

    } else {

      prediction_data <- screening_data %>%
        select(all_of(model_columns))

      names(prediction_data) <- internal_names

      stopifnot(
        all(vapply(prediction_data, is.numeric, logical(1))),
        all(is.finite(as.matrix(prediction_data)))
      )

      # -----------------------------
      # 8. Calculate 95% prediction intervals
      # -----------------------------
      # Fit using low-anomaly training samples only.
      # Apply to high-anomaly training and testing samples.
      set.seed(42)

      interval_model <- rfpi(
        formula = Emission_Mt ~ .,
        traindata = training_data,
        testdata = prediction_data,
        alpha = 0.05,
        calibration = TRUE,
        split_rule = "l1",
        pi_method = "quant",
        rf_package = "rfsrc",
        params_rfsrc = interval_parameters,
        params_calib = list(
          range = c(0.945, 0.955)
        )
      )

      prediction_intervals <- as.data.frame(
        interval_model$quant_interval
      )

      stopifnot(
        nrow(prediction_intervals) == nrow(screening_data),
        all(c("lower", "upper") %in% names(prediction_intervals)),
        all(is.finite(prediction_intervals$lower)),
        all(is.finite(prediction_intervals$upper)),
        all(
          prediction_intervals$lower <=
            prediction_intervals$upper
        )
      )

      # -----------------------------
      # 9. Flag records outside prediction intervals
      # -----------------------------
      interval_results <- interval_results %>%
        mutate(
          lower = prediction_intervals$lower,
          upper = prediction_intervals$upper,
          Potential_high_uncertainty =
            Emission_Mt < lower | Emission_Mt > upper
        )
    }

    flagged_records <- interval_results %>%
      filter(Potential_high_uncertainty)

    # -----------------------------
    # 10. Save intervals and flagged records
    # -----------------------------
    write.csv(
      interval_results,
      paste0(
        "results/", sector, "_", split,
        "_prediction_intervals.csv"
      ),
      row.names = FALSE
    )

    write.csv(
      flagged_records,
      paste0(
        "results/", sector, "_", split,
        "_flagged_records.csv"
      ),
      row.names = FALSE
    )

    all_flagged_records[[
      paste(sector, split, sep = "_")
    ]] <- flagged_records
  }
}

# -----------------------------
# 11. Combine flagged records across seven sectors
# -----------------------------
write.csv(
  bind_rows(all_flagged_records),
  "results/all_sectors_flagged_records.csv",
  row.names = FALSE
)
