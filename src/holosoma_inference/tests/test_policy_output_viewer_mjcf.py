"""Tests for policy_output_viewer._build_dual_g1_mjcf.

2026-05-05 code review blocker #2: verify that the assembled
dual-G1 MJCF actually loads in MuJoCo and that both copies retain
mesh geometry. Catches regressions where a rename pass silently breaks
mesh="..." refs in one of the copies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Import the module via its file path: policy_output_viewer.py lives at
# the top of src/holosoma_inference/ (not inside the nested package),
# so a plain `import` under pytest's sys.path doesn't work uniformly
# across bazel and pip-editable installs.
def _import_viewer_module():
    import importlib.util

    here = Path(__file__).resolve()
    # tests/ is a sibling of policy_output_viewer.py
    viewer_path = here.parent.parent / "policy_output_viewer.py"
    if not viewer_path.is_file():
        pytest.skip(f"policy_output_viewer.py not found at {viewer_path}")
    spec = importlib.util.spec_from_file_location(
        "policy_output_viewer", viewer_path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _find_g1_mjcf():
    try:
        import holosoma_retargeting as _hr  # noqa: F401
    except ImportError:
        pytest.skip("holosoma_retargeting not importable")
    return Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"


@pytest.fixture(scope="module")
def mujoco_mod():
    try:
        import mujoco
    except ImportError:
        pytest.skip("mujoco not importable")
    # Don't spin up a window; validation-only.
    os.environ.setdefault("MUJOCO_GL", "disable")
    return mujoco


@pytest.fixture(scope="module")
def viewer():
    return _import_viewer_module()


@pytest.fixture(scope="module")
def g1_path():
    p = _find_g1_mjcf()
    if not p.exists():
        pytest.skip(f"G1 MJCF missing: {p}")
    return p


def test_dual_mjcf_loads(mujoco_mod, viewer, g1_path, tmp_path):
    """Assembled MJCF round-trips through MuJoCo's loader."""
    combined = viewer._build_dual_g1_mjcf(g1_path, offset=0.8)
    # Write to the G1 asset dir so relative mesh paths resolve.
    tmp = g1_path.parent / "_test_dual_policy_output_viewer.xml"
    try:
        tmp.write_text(combined)
        model = mujoco_mod.MjModel.from_xml_path(str(tmp))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    assert model.njnt > 2, "combined model missing joints"
    assert model.nbody > 2, "combined model missing bodies"


def test_dual_mjcf_has_both_freejoints(mujoco_mod, viewer, g1_path):
    """Each copy gets its own named freejoint (root_cmd / root_act)."""
    combined = viewer._build_dual_g1_mjcf(g1_path, offset=0.8)
    tmp = g1_path.parent / "_test_dual_policy_output_viewer_fj.xml"
    try:
        tmp.write_text(combined)
        model = mujoco_mod.MjModel.from_xml_path(str(tmp))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    free_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco_mod.mjtJoint.mjJNT_FREE:
            free_names.append(model.joint(j).name)
    assert "root_cmd" in free_names, f"missing root_cmd freejoint: {free_names}"
    assert "root_act" in free_names, f"missing root_act freejoint: {free_names}"


def test_dual_mjcf_retains_mesh_geoms_in_both_copies(mujoco_mod, viewer, g1_path):
    """Regression test for review blocker #2.

    If the rename pass ever touches mesh="..." refs or duplicates the
    <asset> block with suffixes, one (or both) copies will end up with
    broken mesh resolution. MuJoCo usually still loads with a warning
    in that case, so assert that both sides have at least one mesh
    geom each, and that every mesh geom references a real mesh asset
    in the model.
    """
    combined = viewer._build_dual_g1_mjcf(g1_path, offset=0.8)
    tmp = g1_path.parent / "_test_dual_policy_output_viewer_mesh.xml"
    try:
        tmp.write_text(combined)
        model = mujoco_mod.MjModel.from_xml_path(str(tmp))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    cmd_mesh_count = 0
    act_mesh_count = 0
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco_mod.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[g])
        assert 0 <= mesh_id < model.nmesh, (
            f"geom {g} has invalid mesh id {mesh_id} "
            "(nmesh={model.nmesh}); suffix pass likely broke a mesh ref"
        )
        parent_body = int(model.geom_bodyid[g])
        body_name = model.body(parent_body).name
        if body_name.endswith("_cmd"):
            cmd_mesh_count += 1
        elif body_name.endswith("_act"):
            act_mesh_count += 1
    assert cmd_mesh_count > 0, "_cmd copy has no mesh geoms"
    assert act_mesh_count > 0, "_act copy has no mesh geoms"


def test_dual_mjcf_does_not_suffix_mesh_assets(viewer, g1_path):
    """The source MJCF's mesh names (e.g. `pelvis`) must survive
    unchanged in the combined XML. If a rename pass starts touching
    <asset>, mesh="pelvis" geom refs will fall through to the
    `pelvis_cmd` / `pelvis_act` rename and resolution breaks.
    """
    combined = viewer._build_dual_g1_mjcf(g1_path, offset=0.8)
    assert '<mesh name="pelvis"' in combined, (
        "dual MJCF missing bare 'pelvis' mesh asset; the <asset> block "
        "appears to have been rewritten, which will break mesh refs in "
        "the copies (review blocker #2)"
    )
    # And the copies still reference the bare name:
    assert 'mesh="pelvis"' in combined
