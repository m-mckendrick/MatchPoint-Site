import pandas as pd
from pathlib import Path

from .match_context import add_pre_match_context_features
from .match_results import add_match_result_features
from .serve_return import add_serve_return_features
from .pressure_performance import add_pressure_performance_features
from .fatigue import add_fatigue_index
from .form import add_form_features

from src.feature_engineering.form_profiles import (
    add_form_all,
    add_form_same,
    add_h2h_form_all,
    add_h2h_form_same,
    add_tournament_form_all,
)


from data.loader import load_matches, PROCESSED_PATH

import time
from rich.console import Console

console = Console()

"""
feature config
These toggles let you turn features on/off without editing code.
This is extremely useful on a VM because:
we can disable form and make everything else is stable
we can run "light mode" for debugging
"""

FEATURE_CONFIG = {
    "pre_match": True,
    "serve_return": True,
    "match_result": True,
    "fatigue": True,
    "pressure": True,

    "form": True,
    "surface_form": True,
    "h2h": True,
    "tournament_form": True,
}

"""
just a function to help display what has run and the outcome
"""
def timed_step(name, func, df):
    console.log(f"Running: {name}/")
    start = time.time()
    df = func(df)
    console.log(f"Finished Running {name} completed in {time.time() - start:.2f}s/")
    return df


# df acts as both the synthetic row and all the processed matches
#full df is always the full set of processed matches
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)

    # light features
    if FEATURE_CONFIG["pre_match"]:
        df = timed_step("Pre-match context", add_pre_match_context_features, df)

    if FEATURE_CONFIG["serve_return"]:
        df = timed_step("Serve/Return features", add_serve_return_features, df)

    if FEATURE_CONFIG["match_result"]:
        df = timed_step("Match result features", add_match_result_features, df)

    if FEATURE_CONFIG["fatigue"]:
        df = timed_step("Fatigue index", add_fatigue_index, df)

    if FEATURE_CONFIG["pressure"]:
        df = timed_step("Pressure performance", add_pressure_performance_features, df)

    # form features the heavy ones

    if FEATURE_CONFIG["form"]:
        df = timed_step(
            "Short-term form",
            add_form_all,
            df
        )
        df = timed_step(
            "Short-term surface form",
            add_form_same,
            df
        )

    if FEATURE_CONFIG["h2h"]:
        df = timed_step(
            "H2H form",
            add_h2h_form_all,
            df
        )
        df = timed_step(
            "H2H surface form",
            add_h2h_form_same,
            df
        )


    if FEATURE_CONFIG["tournament_form"]:
        df = timed_step(
            "Tournament form",
            add_tournament_form_all,
            df
        )

    return df


if __name__ == "__main__":

    output_path = PROCESSED_PATH / f"features.csv"

    df=load_matches()

    TEST_MODE = True
    TEST_ROWS = 500

    if TEST_MODE:
        print(f"\n[TEST MODE] Using only the first {TEST_ROWS} rows for debugging.\n")
        df = df.head(TEST_ROWS).copy()

    df = build_features(df)

    df.to_csv(output_path, index=False)

    print("\nsize:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("column names: ")
    for col in df.columns:
        print(col)

    print("\nDone!")


