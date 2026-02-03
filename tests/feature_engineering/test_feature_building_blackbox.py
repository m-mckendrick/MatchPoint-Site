import pandas as pd
from src.feature_engineering.feature_builder import build_features

#test combining csvs
def test_build_features_combines_multiple_csvs(tmp_path, cleaned_match_df):
    #write two separate cleaned csv's
    (tmp_path / "2020.csv").write_text(cleaned_match_df.to_csv(index=False))
    (tmp_path / "2021.csv").write_text(cleaned_match_df.to_csv(index=False))

    out = build_features(processed_dir=tmp_path)

    assert len(out) == len(cleaned_match_df) * 2

#testing the ml model add label = 1
def test_build_features_adds_label_column(tmp_path, cleaned_match_df):
    (tmp_path / "2020.csv").write_text(cleaned_match_df.to_csv(index=False))

    out = build_features(processed_dir=tmp_path)

    assert "label" in out.columns
    assert set(out["label"].unique()) == {1}

#is added features present
def test_build_features_adds_expected_feature_columns(tmp_path, cleaned_match_df):
    (tmp_path / "2020.csv").write_text(cleaned_match_df.to_csv(index=False))

    out = build_features(processed_dir=tmp_path)

    #only some of added columns
    assert "seed_diff" in out.columns
    assert "rank_diff" in out.columns
    assert "w_first_serve_pct" in out.columns
    assert "w_second_serve_won_pct" in out.columns
