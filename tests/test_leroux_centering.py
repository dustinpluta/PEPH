import numpy as np

from peph.model.leroux_objective import project_center_by_component


def test_project_center_by_component_weighted() -> None:
    # two components: {0,1,2} and {3,4}
    comp = np.array([0, 0, 0, 1, 1], dtype=int)
    u = np.array([1.0, 2.0, 3.0, 10.0, 20.0], dtype=float)
    w = np.array([1.0, 2.0, 1.0, 1.0, 3.0], dtype=float)

    uc = project_center_by_component(u, comp, w)

    # weighted mean per component should be 0
    for c in [0, 1]:
        idx = np.where(comp == c)[0]
        m = float(np.sum(w[idx] * uc[idx]) / np.sum(w[idx]))
        assert abs(m) < 1e-12