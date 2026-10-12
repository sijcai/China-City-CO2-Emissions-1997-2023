# Random Forest point prediction across seven sectors
# This script predicts emissions for potentially high-uncertainty records and city-year observations with missing emissions.

library(dplyr)
library(mlr3)
library(mlr3tuning)
library(mlr3learners)
library(ranger)

# Required input files:
# {sector}_train_lof.csv
# {sector}_test_lof.csv
# {sector}_unobserved.csv
# results/{sector}_rf_model.rds
# results/all_sectors_flagged_records.csv

# {sector} refers to:
# petroleum_coal_other_fuels, chemical_products, building_materials, iron_and_steel, nonferrous_metals, electricity_and_heat, other.

# Required columns:
# City_id, Year, and the sector-specific predictors below.

# -----------------------------
# 1. Define sector-specific predictors
# -----------------------------
sector_predictors <- list(
  petroleum_coal_other_fuels = c(
    "Emp-PCFPI", "RCoal", "Coke", "CrudeOil",
    "ImpArea", "PopDensity", "CoalCap"
  ),

  chemical_products = c(
    "Emp-Chem", "SynAmmonia", "CPlantCap", "Ethylene",
    "WaterArea", "ImpArea", "AvgHumidity", "MG_Supply"
  ),

  building_materials = c(
    "Cement", "GreenArea", "ForestArea", "LGExp"
  ),

  iron_and_steel = c(
    "PIron", "CSteel", "Emp-ISI", "ImpArea", "AvgTemp"
  ),

  nonferrous_metals = c(
    "PrimAlum", "Emp-NFMSI", "Output-NFMSI",
    "GrassArea", "CropArea", "ImpArea", "SewageDis"
  ),

  electricity_and_heat = c(
    "CoalCap", "Output-EHS", "ImpArea",
    "ProdCap", "WaterSupply", "WaterArea"
  ),

  other = c(
    "BuiltArea", "GreenArea", "WaterPipeLen",
    "GDP", "PopDensity"
  )
)

flagged_records <- read.csv(
  "results/all_sectors_flagged_records.csv"
)

for (sector in names(sector_predictors)) {

  predictors <- sector_predictors[[sector]]

  # -----------------------------
  # 2. Read the fitted model
  # -----------------------------
  fitted_model <- readRDS(
    paste0("results/", sector, "_rf_model.rds")
  )

  # -----------------------------
  # 3. Select potentially high-uncertainty records
  # -----------------------------
  train_data <- read.csv(
    paste0(sector, "_train_lof.csv"),
    check.names = FALSE
  )

  test_data <- read.csv(
    paste0(sector, "_test_lof.csv"),
    check.names = FALSE
  )

  uncertain_ids <- flagged_records %>%
    filter(Sector == sector) %>%
    distinct(City_id, Year)

  uncertain_data <- bind_rows(train_data, test_data) %>%
    semi_join(
      uncertain_ids,
      by = c("City_id", "Year")
    )

  # -----------------------------
  # 4. Read the prediction set
  # -----------------------------
  prediction_set <- read.csv(
    paste0(sector, "_unobserved.csv"),
    check.names = FALSE
  )

  application_data <- list(
    uncertain = uncertain_data,
    unobserved = prediction_set
  )

  # -----------------------------
  # 5. Predict emissions for both datasets
  # -----------------------------
  for (dataset in names(application_data)) {

    data <- application_data[[dataset]]

    predictor_data <- data %>%
      select(all_of(predictors))

    # Use the same internal predictor names as script 02.
    names(predictor_data) <- paste0(
      "X",
      seq_along(predictors)
    )

    data$Predicted_emission_Mt <- numeric(nrow(data))

    if (nrow(data) > 0) {
      data$Predicted_emission_Mt <-
        fitted_model$predict_newdata(
          newdata = predictor_data
        )$response
    }

    # -----------------------------
    # 6. Save point predictions
    # -----------------------------
    write.csv(
      data,
      paste0(
        "results/", sector, "_", dataset,
        "_point_predictions.csv"
      ),
      row.names = FALSE
    )
  }
}
