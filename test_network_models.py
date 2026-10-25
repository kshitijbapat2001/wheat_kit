import numpy as np
import pytest

pytest.importorskip("transformers")
pytestmark = pytest.mark.network


def test_segment_organs_runs_and_returns_gwfss_classes(green_on_soil):
    from wheatkit.tasks import canopy_cover, segment_organs

    seg = segment_organs(green_on_soil)
    assert seg.classes == {0: "background", 1: "head", 2: "stem", 3: "leaf"}
    assert seg.mask.shape == green_on_soil.shape[:2]
    assert abs(sum(seg.fractions.values()) - 1) < 1e-6
    assert 0 <= canopy_cover(green_on_soil, method="segformer").cover <= 1
    assert np.issubdtype(seg.mask.dtype, np.integer)
