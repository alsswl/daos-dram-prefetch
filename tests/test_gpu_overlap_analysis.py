from analyze_gpu_overlap import merge, Intersections


def test_merge_does_not_double_count_overlapping_compute_kernels():
    assert merge([(0,4),(2,6),(6,8),(10,11),(20,20)]) == [(0,8),(10,11)]


def test_partial_intersections_and_empty_regions():
    spans = Intersections([(0,4),(2,6),(10,12)])
    assert spans.overlap(3,11) == 4
    assert spans.overlap(6,10) == 0
    assert spans.overlap(-1,20) == 8
