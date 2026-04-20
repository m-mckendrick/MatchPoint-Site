import pandas as pd
from pathlib import Path

from .match_context import add_pre_match_context_features
from .match_results import add_match_result_features
from .serve_return import add_serve_return_features
from .pressure_performance import add_pressure_performance_features
from .fatigue import add_fatigue_index
from .form import add_form_features

from src.feature_engineering.form_profiles import (
    add_career_form,
    add_h2h_form,
    add_52week_form,
    add_tournament_form,
)


from data.loader import load_matches

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

    "form": False,
    "surface_form": False,
    "h2h": False,
    "career": False,
    "recency": False,
    "tournament_form": False,
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
            lambda d: add_form_features(d, windows=[3], surface_mode="all"),
            df
        )

    if FEATURE_CONFIG["career"]:
        df = timed_step(
            "Career form",
            lambda d: add_career_form(d, windows=[10], surface_mode="all"),
            df
        )

    if FEATURE_CONFIG["h2h"]:
        df = timed_step(
            "H2H form",
            lambda d: add_h2h_form(d, windows=[3], surface_mode="all"),
            df
        )

    if FEATURE_CONFIG["recency"]:
        df = timed_step(
            "52-week form",
            lambda d: add_52week_form(d, windows=[5], surface_mode="all"),
            df
        )

    if FEATURE_CONFIG["tournament_form"]:
        df = timed_step(
            "Tournament form",
            lambda d: add_tournament_form(d, windows=[5], surface_mode="all"),
            df
        )

    # --- SURFACE FORM ---
    if FEATURE_CONFIG["surface_form"]:
        df = timed_step(
            "Surface form",
            lambda d: add_form_features(d, windows=[5], surface_mode="same"),
            df
        )

    return df


    """
    
    full_na_audit(df)
    
    full_na_audit(df)
    

              # short-term
    df = add_form_features(df, windows=[2], surface_mode="all")
    df = add_career_form(df, windows=[10], surface_mode="all")           # longer-term
    df = add_h2h_form(df, windows=[3], surface_mode="all")               # rivalry
    df = add_52week_form(df, windows=[5], surface_mode="all")            # recency
    df = add_tournament_form(df, windows=[5], surface_mode="all")

    df = add_form_features(df, windows=[5], surface_mode="same")         # short-term surface form
    df = add_career_form(df, windows=[10], surface_mode="same")          # long-term surface form
    df = add_h2h_form(df, windows=[3], surface_mode="same")              # rivalry on this surface
    df = add_52week_form(df, windows=[5], surface_mode="same")           # recency on this surface
    df = add_tournament_form(df, windows=[5], surface_mode="same")

    
    """

    return df


if __name__ == "__main__":


    df=load_matches()
    df = build_features(df)

    print("\nsize:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("column names: ")
    for col in df.columns:
        print(col)

    print("\nDone!")


