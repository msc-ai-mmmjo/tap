from gradio_demo import constants


def test_model_and_name_are_set():
    assert isinstance(constants.MODEL, str) and "/" in constants.MODEL
    assert isinstance(constants.MODEL_NAME, str) and constants.MODEL_NAME


def test_heatmap_data_alias_resolves():
    sample: constants.HeatmapData = [("hello", 0.5), ("world", None), ("!", "label")]
    assert sample[0] == ("hello", 0.5)
    assert sample[1] == ("world", None)
    assert sample[2] == ("!", "label")
