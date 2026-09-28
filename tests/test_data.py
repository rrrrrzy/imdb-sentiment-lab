"""Protect the experiment from leakage and silent collection failures."""
import pandas as pd
import pytest

from sentiment_lab.config import Paths
from sentiment_lab.data.collect import collect, discover_archive
from sentiment_lab.data.prepare import audit_and_clean, clean_text


def test_cleaning_retains_negation_and_decodes_entities():
    assert clean_text("I am NOT<br /> happy &amp; satisfied!  ") == "i am not happy & satisfied!"


def test_conflicts_duplicates_and_test_overlap_are_removed():
    rows = [
        ("a", "train", 1, "<b>Great</b> film"),
        ("b", "train", 1, "great film"),
        ("c", "train", 0, "ambiguous film"),
        ("d", "train", 1, "ambiguous film"),
        ("e", "test", 1, "Great film"),
        ("f", "test", 0, "bad film"),
        ("g", "test", 0, "<br>"),
    ]
    frame, audit = audit_and_clean(pd.DataFrame(rows, columns=["id", "split", "label", "text"]))
    assert set(frame.id) == {"a", "f"}
    assert audit["empty_removed"] == 1
    assert audit["cross_split_test_duplicates_removed"] == 1
    assert audit["within_split"]["train"]["conflicting_rows_removed"] == 2
    assert audit["within_split"]["train"]["duplicate_rows_removed"] == 1
    assert not set(frame.loc[frame.split == "train", "text_hash"]) & set(frame.loc[frame.split == "test", "text_hash"])


def test_download_url_is_discovered_from_publisher_html():
    assert discover_archive('<a href="aclImdb_v1.tar.gz">archive</a>').endswith("/sentiment/aclImdb_v1.tar.gz")
    with pytest.raises(ValueError, match="off-site"):
        discover_archive('<a href="https://evil.example/aclImdb_v1.tar.gz">archive</a>')


def test_offline_mode_never_needs_a_network_request(tmp_path):
    with pytest.raises(FileNotFoundError, match="Offline mode"):
        collect(Paths(tmp_path), offline=True)
