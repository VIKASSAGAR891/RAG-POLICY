def test_project_smoke():
    import src.policy_qa.pipeline
    import src.policy_qa.ingestion
    import src.policy_qa.chunker
    assert True


def test_settings_root_is_repository_root():
    from config.settings import ROOT, SETTINGS

    assert ROOT == ROOT.parent / "POLICY-RAG"
    assert SETTINGS.raw_data_dir == ROOT / "data" / "raw"
