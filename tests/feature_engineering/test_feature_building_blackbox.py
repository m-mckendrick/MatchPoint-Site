import pandas as pd
from src.feature_engineering.feature_builder import build_features

#test combining csvs
def test_build_features_combines_multiple_csvs(tmp_path, p1p2_match_df):
    #write two separate cleaned csv's
    (tmp_path / "2020.csv").write_text(p1p2_match_df.to_csv(index=False))
    (tmp_path / "2021.csv").write_text(p1p2_match_df.to_csv(index=False))

    out = build_features(processed_dir=tmp_path)

    assert len(out) == len(p1p2_match_df) * 2

#is added features present
def test_build_features_adds_expected_feature_columns(tmp_path, p1p2_match_df):
    (tmp_path / "2020.csv").write_text(p1p2_match_df.to_csv(index=False))

    out = build_features(processed_dir=tmp_path)

    #only some of added columns
    assert "rank_diff" in out.columns
    assert "p1_first_serve_pct" in out.columns
    assert "p1_second_serve_won_pct" in out.columns
