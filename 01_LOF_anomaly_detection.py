# LOF screening of city-year emission records across seven sectors
# This script documents the procedure used to classify activity-based emission records into low-anomaly and high-anomaly samples.

import numpy as np
import pandas as pd
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler

# Required input files:
# {sector}_train.csv and {sector}_test.csv
# {sector} refers to one of the following sector identifiers:
# petroleum_coal_other_fuels, chemical_products, building_materials, iron_and_steel, nonferrous_metals, electricity_and_heat, other.
# Example:
# petroleum_coal_other_fuels_train.csv
# petroleum_coal_other_fuels_test.csv

# Required columns:
# City_id, Year, Emission_Mt, and the sector-specific predictors below.

# -----------------------------
# 1. Define sector-specific predictors and LOF parameters
# -----------------------------
# Sector names follow the manuscript.
# Parameters follow the supplied research scripts.
sector_settings = {
    "petroleum_coal_other_fuels": {
        "predictors": [
            "Output-PCFPI",
            "CrudeOil",
            "Coke",
            "RCoal"
        ],
        "contamination": 0.15,
        "n_neighbors": 25
    },

    "chemical_products": {
        "predictors": [
            "Output-Chem",
            "CCarbide",
            "ChemFiber",
            "SodaAsh",
            "SynAmmonia",
            "Ethylene"
        ],
        "contamination": 0.20,
        "n_neighbors": 20
    },

    "building_materials": {
        "predictors": [
            "Output-BM",
            "Cement"
        ],
        "contamination": 0.25,
        "n_neighbors": 25
    },

    "iron_and_steel": {
        "predictors": [
            "Output-ISI",
            "PIron",
            "CSteel"
        ],
        "contamination": 0.25,
        "n_neighbors": 20
    },

    "nonferrous_metals": {
        "predictors": [
            "Output-NFMSI",
            "PrimAlum"
        ],
        "contamination": 0.20,
        "n_neighbors": 25
    },

    "electricity_and_heat": {
        "predictors": [
            "Output-EHS",
            "CoalCap"
        ],
        "contamination": 0.25,
        "n_neighbors": 25
    },

    "other": {
        "predictors": [
            "NTL",
            "BuiltArea",
            "IndGDP"
        ],
        "contamination": 0.25,
        "n_neighbors": 25
    }
}

for sector, settings in sector_settings.items():

    # -----------------------------
    # 2. Read training and testing data
    # -----------------------------
    train_data = pd.read_csv(f"{sector}_train.csv")
    test_data = pd.read_csv(f"{sector}_test.csv")

    # LOF screening uses predictors and activity-based emissions jointly.
    lof_variables = settings["predictors"] + ["Emission_Mt"]

    # -----------------------------
    # 3. Normalize variables
    # -----------------------------
    # Fit the scaler using training data only.
    # Transform testing data using the same fitted scaler.
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_data[lof_variables]
    )
    test_scaled = scaler.transform(
        test_data[lof_variables]
    )

    # -----------------------------
    # 4. Fit the LOF model using training data
    # -----------------------------
    lof_model = LOF(
        contamination=settings["contamination"],
        n_neighbors=settings["n_neighbors"]
    )

    lof_model.fit(train_scaled)

    # -----------------------------
    # 5. Calculate LOF scores and classify samples
    # -----------------------------
    train_data["score_lof"] = lof_model.decision_scores_
    test_data["score_lof"] = lof_model.decision_function(
        test_scaled
    )

    # Apply the threshold derived from training data to both datasets.
    threshold = lof_model.threshold_

    train_data["label_lof"] = np.where(
        train_data["score_lof"] > threshold,
        "high_anomaly",
        "low_anomaly"
    )

    test_data["label_lof"] = np.where(
        test_data["score_lof"] > threshold,
        "high_anomaly",
        "low_anomaly"
    )

    # -----------------------------
    # 6. Save screening results
    # -----------------------------
    train_data.to_csv(
        f"{sector}_train_lof.csv",
        index=False
    )

    test_data.to_csv(
        f"{sector}_test_lof.csv",
        index=False
    )
