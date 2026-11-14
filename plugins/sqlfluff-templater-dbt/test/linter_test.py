"""The Test file for the linter class."""

import os
import os.path
import shutil
from pathlib import Path
from unittest import mock

import pytest

from sqlfluff.cli.commands import lint
from sqlfluff.core import FluffConfig, Linter
from sqlfluff.core.linter import runner
from sqlfluff.core.linter.common import DeferredRenderTask
from sqlfluff.utils.testing.cli import invoke_assert_code


def _write_config(path: Path, contents: str) -> None:
    """Write a test configuration file."""
    path.write_text(contents, encoding="utf-8")


@pytest.mark.parametrize(
    "path", ["models/my_new_project/disabled_model.sql", "macros/echo.sql"]
)
def test__linter__skip_file(path, project_dir, dbt_fluff_config):
    """Test that the linter skips disabled dbt models and macros."""
    conf = FluffConfig(configs=dbt_fluff_config)
    lntr = Linter(config=conf)
    model_file_path = os.path.join(project_dir, path)
    linted_path = lntr.lint_path(path=model_file_path)
    # Check that the file is still there
    assert len(linted_path.files) == 1
    linted_file = linted_path.files[0]
    # Normalise paths to control for OS variance
    assert os.path.normpath(linted_file.path) == os.path.normpath(model_file_path)
    assert not linted_file.templated_file
    assert not linted_file.tree


def test__linter__lint_ephemeral_3_level(project_dir, dbt_fluff_config):
    """Test linter can lint a project with 3-level ephemeral dependencies."""
    # This was previously crashing inside dbt, in a function named
    # inject_ctes_into_sql(). (issue 2671).
    conf = FluffConfig(configs=dbt_fluff_config)
    lntr = Linter(config=conf)
    model_file_path = os.path.join(project_dir, "models/ephemeral_3_level")
    lntr.lint_path(path=model_file_path)


def test__linter__config_pairs(dbt_fluff_config):
    """Test that the dbt templater returns version information in it's config."""
    conf = FluffConfig(configs=dbt_fluff_config)
    lntr = Linter(config=conf)
    # NOTE: This method is called within the config readout.
    assert lntr.templater.config_pairs() == [
        ("templater", "dbt"),
        ("dbt", lntr.templater.dbt_version),
    ]


def test_dbt_target_dir(tmpdir, dbt_project_folder, profiles_dir):
    """Test with dbt project in subdir that target/ is created in the correct place.

    https://github.com/sqlfluff/sqlfluff/issues/2895
    """
    tmp_base_dir = str(tmpdir)
    tmp_dbt_dir = os.path.join(tmp_base_dir, "dir1", "dir2", "dbt")
    tmp_project_dir = os.path.join(tmp_dbt_dir, "dbt_project")
    os.makedirs(os.path.dirname(tmp_dbt_dir))
    shutil.copytree(
        dbt_project_folder,
        tmp_dbt_dir,
    )
    os.unlink(os.path.join(tmp_dbt_dir, ".sqlfluff"))
    old_cwd = os.getcwd()
    # Invoke SQLFluff from <<tmpdir>>, linting a file in the dbt project at
    # <<tmp_project_dir>>/dir1/dir2/dbt/dbt_project. Prior to the bug fix, a
    # "target" directory would incorrectly be created in <<tmp_project_dir>>.
    # (It should be created in <<tmp_project_dir>>/dir1/dir2/dbt/dbt_project.)
    os.chdir(tmp_base_dir)
    with open(".sqlfluff", "w") as f:
        print(
            f"""[sqlfluff]
templater = dbt
dialect = postgres

[sqlfluff:templater:dbt]
project_dir = {tmp_project_dir}
profiles_dir = {old_cwd}/{profiles_dir}
""",
            file=f,
        )
    try:
        invoke_assert_code(
            ret_code=0,
            args=[
                lint,
                [
                    "dir1/dir2/dbt/dbt_project/models/my_new_project/use_dbt_utils.sql",
                ],
            ],
        )
        assert not os.path.exists("target")
        assert os.path.exists("dir1/dir2/dbt/dbt_project/target")
    finally:
        os.chdir(old_cwd)


def test__dbt_templater__templates_in_worker_false():
    """DbtTemplater.templates_in_worker must be False.

    The dbt templater holds a compiled dbt manifest in memory that is not
    safe to rebuild inside a worker process. Parallel runners check this flag
    and fall back to the main-process (fat-path) templating when it is False,
    sending only the pre-rendered RenderedFile across the IPC boundary.
    """
    from sqlfluff_templater_dbt.templater import DbtTemplater

    assert DbtTemplater.templates_in_worker is False


def test__dbt_linter__parallel_partials_path(project_dir, dbt_fluff_config):
    """Parallel linting with the dbt templater uses the main-process render path.

    Because DbtTemplater.templates_in_worker=False, ParallelRunner.iter_partials
    must yield callable partials (RenderedFile already attached), NOT
    DeferredRenderTask objects. This ensures the manifest is only compiled once
    in the main process and the worker only handles the lint step.
    """
    conf = FluffConfig(configs=dbt_fluff_config)
    lntr = Linter(config=conf)
    models_dir = os.path.join(project_dir, "models", "my_new_project")
    files = [
        os.path.join(models_dir, "operator_errors.sql"),
        os.path.join(models_dir, "single_trailing_newline.sql"),
    ]

    thd_runner = runner.MultiThreadRunner(lntr, conf, processes=2)
    partials = list(thd_runner.iter_partials(files, fix=False))

    assert len(partials) == 2
    for _fname, task in partials:
        # Must be a callable partial, not a DeferredRenderTask.
        assert callable(task)
        assert not isinstance(task, DeferredRenderTask)

    # Also verify that the full parallel lint run completes without error.
    results = list(thd_runner.run(files, fix=False))
    assert len(results) == 2


@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__dbt_linter__root_templater_reused(
    set_relations_cache, project_dir, dbt_fluff_config
):
    """The root dbt templater renders files, and is reused across operations.

    Its dbt project is only loaded once, and stays loaded until it's closed.
    """
    from dbt.adapters.factory import FACTORY

    lntr = Linter(config=FluffConfig(configs=dbt_fluff_config))
    root_templater = lntr.templater
    fname = os.path.join(
        project_dir, "models/my_new_project/single_trailing_newline.sql"
    )
    with mock.patch.object(
        root_templater, "process", wraps=root_templater.process
    ) as process:
        for _ in range(2):
            result = lntr.lint_paths((fname,))
            assert len(result.paths[0].files) == 1
            assert lntr.templater is root_templater
            assert set(FACTORY.adapters.values()) == {root_templater._adapter}
    assert process.call_count == 2
    # The manifest was loaded once, by the first operation.
    manifest = root_templater.dbt_manifest
    lntr.lint_paths((fname,))
    assert root_templater.dbt_manifest is manifest
    root_templater.close()
    assert not FACTORY.adapters


def _make_dbt_projects(tmp_path, dbt_project_folder):
    """Make two dbt projects, nested in a project using the jinja templater.

    Returns the root, a jinja file, the dbt files, and their rendered SQL.
    """
    root = tmp_path / "mixed_projects"
    root.mkdir()
    jinja_file = root / "jinja.sql"
    jinja_file.write_text("select 1 from {{ table_name }}\n", encoding="utf-8")
    _write_config(
        root / ".sqlfluff",
        """[sqlfluff]
dialect = postgres
templater = jinja

[sqlfluff:templater:jinja:context]
table_name = table_a
""",
    )

    profiles_dir = Path(dbt_project_folder, "profiles_yml").resolve()
    dbt_files = []
    expected_rendered = {}
    for project_index, project_name in enumerate(("project_a", "project_b"), start=1):
        project_root = root / project_name
        shutil.copytree(Path(dbt_project_folder, "dbt_project"), project_root)
        _write_config(
            project_root / ".sqlfluff",
            f"""[sqlfluff]
dialect = postgres
templater = dbt

[sqlfluff:templater:dbt]
profiles_dir = {profiles_dir}
project_dir = {project_root}

[sqlfluff:templater:dbt:context]
passed_through_cli = {project_index}
""",
        )
        dbt_file = str(project_root / "models/vars_from_cli.sql")
        dbt_files.append(dbt_file)
        expected_rendered[dbt_file] = f"-- Issue #1262\nSELECT {project_index}\n"
    return root, jinja_file, dbt_files, expected_rendered


def _rendered_files(result):
    """Return the templated SQL of each linted file, by path."""
    return {
        linted_file.path: linted_file.templated_file.templated_str
        for linted_path in result.paths
        for linted_file in linted_path.files
    }


@pytest.mark.parametrize("processes", [1, 2])
@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__linter__supports_nested_dbt_projects(
    set_relations_cache, tmp_path, dbt_project_folder, processes
):
    """Lint built-in and independent nested dbt templaters together."""
    from dbt.adapters.factory import FACTORY

    root, jinja_file, dbt_files, expected_rendered = _make_dbt_projects(
        tmp_path, dbt_project_folder
    )
    linter = Linter(config=FluffConfig.from_path(str(root)))
    result = linter.lint_paths(
        (str(jinja_file), *dbt_files),
        processes=processes,
    )

    assert not [violation for violation in result.get_violations() if violation.fatal]
    assert _rendered_files(result) == {
        str(jinja_file): "select 1 from table_a\n",
        **expected_rendered,
    }

    reversed_result = linter.lint_paths(tuple(reversed(dbt_files)), processes=processes)
    reversed_files = _rendered_files(reversed_result)
    assert list(reversed_files) == list(reversed(dbt_files))
    assert reversed_files == expected_rendered
    # No dbt adapter is left registered.
    assert not FACTORY.adapters


@pytest.mark.parametrize("processes", [1, 2])
@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__linter__dbt_inline_config_selects_project_config(
    set_relations_cache, tmp_path, dbt_project_folder, processes, monkeypatch
):
    """Inline config can change the dbt config of a file.

    The file is deferred, and rendered by another dbt templater, which loads
    the project with the inline config, after the other files. Each file is
    read once.
    """
    from sqlfluff_templater_dbt.templater import DbtTemplater

    root, _, dbt_files, expected_rendered = _make_dbt_projects(
        tmp_path, dbt_project_folder
    )
    models = Path(dbt_files[0]).parent
    inline_file = models / "inline_vars_from_cli.sql"
    inline_file.write_text(
        "-- sqlfluff:templater:dbt:context:passed_through_cli:7\n"
        "SELECT {{ var('passed_through_cli') }}\n",
        encoding="utf-8",
    )
    reads = []
    load_raw_file = Linter._load_raw_file

    def record_read(fname, file_config):
        reads.append(fname)
        return load_raw_file(fname, file_config)

    renders = []
    process = DbtTemplater.process

    def record_render(self, **kwargs):
        renders.append(kwargs["fname"])
        return process(self, **kwargs)

    monkeypatch.setattr(Linter, "_load_raw_file", staticmethod(record_read))
    monkeypatch.setattr(DbtTemplater, "process", record_render)
    linter = Linter(config=FluffConfig.from_path(str(root)))
    result = linter.lint_paths(
        (dbt_files[0], str(inline_file), dbt_files[1]), processes=processes
    )

    assert not [violation for violation in result.get_violations() if violation.fatal]
    assert sorted(reads) == sorted((*dbt_files, str(inline_file)))
    assert renders[-1] == str(inline_file)
    assert _rendered_files(result) == {
        **expected_rendered,
        str(inline_file): (
            "-- sqlfluff:templater:dbt:context:passed_through_cli:7\nSELECT 7\n"
        ),
    }


@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__linter__parse_path_nested_dbt_projects(
    set_relations_cache, tmp_path, dbt_project_folder
):
    """Parse files of several nested dbt projects."""
    from dbt.adapters.factory import FACTORY

    root, jinja_file, dbt_files, expected_rendered = _make_dbt_projects(
        tmp_path, dbt_project_folder
    )
    linter = Linter(config=FluffConfig.from_path(str(root)))
    parsed = {
        parsed.fname: parsed
        for parsed in linter.parse_path(str(root))
        if parsed.fname in (str(jinja_file), *dbt_files)
    }

    assert len(parsed) == 3
    for fname, parsed_file in parsed.items():
        assert not [v for v in parsed_file.violations if v.rule_code() == "TMP"]
        if fname in expected_rendered:
            assert (
                parsed_file.parsed_variants[0].templated_file.templated_str
                == expected_rendered[fname]
            )
    assert not FACTORY.adapters


@pytest.mark.parametrize("assigned", [False, True])
@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__linter__root_dbt_templater_with_nested_dbt_project(
    set_relations_cache, tmp_path, dbt_project_folder, assigned
):
    """A root dbt templater and a nested dbt templater lint files together.

    Whichever templater renders a file loads its project, unloading the other.
    """
    from dbt.adapters.factory import FACTORY

    _, _, dbt_files, expected_rendered = _make_dbt_projects(
        tmp_path, dbt_project_folder
    )
    project_a = str(Path(dbt_files[0]).parents[1])
    lntr = Linter(config=FluffConfig.from_path(project_a))
    if assigned:
        lntr.templater = FluffConfig.from_path(project_a).get_templater()
    root_templater = lntr.templater

    for files in (dbt_files, list(reversed(dbt_files))):
        result = lntr.lint_paths(tuple(files))
        assert not [v for v in result.get_violations() if v.fatal]
        assert _rendered_files(result) == expected_rendered
        assert lntr.templater is root_templater
    # The nested templater was released, and the root one may stay loaded.
    assert set(FACTORY.adapters.values()) <= {root_templater._adapter}
    root_templater.close()
    assert not FACTORY.adapters


@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__dbt_linter__parallel_run_closed_early(
    set_relations_cache, project_dir, dbt_fluff_config
):
    """A parallel run can be stopped early, and dbt can be loaded again.

    The dbt project may be loaded in a pool thread, and unloaded in the main
    thread.
    """
    from dbt.adapters.factory import FACTORY

    from sqlfluff_templater_dbt import templater as dbt_templater_module

    lntr = Linter(config=FluffConfig(configs=dbt_fluff_config))
    fnames = [
        os.path.join(project_dir, "models/my_new_project", name)
        for name in ("single_trailing_newline.sql", "use_var.sql")
    ]
    parallel_runner, _ = runner.get_runner(lntr, lntr.config, processes=2)
    linted = parallel_runner.run(fnames, fix=False)
    next(linted)
    linted.close()
    lntr.templater.close()
    assert not FACTORY.adapters

    assert dbt_templater_module._ADAPTER_LOCK.acquire(timeout=10)
    dbt_templater_module._ADAPTER_LOCK.release()
    result = lntr.lint_paths(tuple(fnames))
    assert len(_rendered_files(result)) == 2


@mock.patch("dbt.adapters.postgres.impl.PostgresAdapter.set_relations_cache")
def test__dbt_linter__same_project_in_threads(
    set_relations_cache, project_dir, dbt_fluff_config
):
    """Linters in several threads can lint the same dbt project at once."""
    import threading

    fnames = tuple(
        os.path.join(project_dir, "models/my_new_project", name)
        for name in ("single_trailing_newline.sql", "use_var.sql", "AM03_test.sql")
    )
    linters = [Linter(config=FluffConfig(configs=dbt_fluff_config)) for _ in range(2)]
    results = {}
    errors = []

    def lint(index):
        try:
            results[index] = linters[index].lint_paths(fnames)
        except Exception as err:  # pragma: no cover
            errors.append(err)

    threads = [threading.Thread(target=lint, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    assert not errors
    for result in results.values():
        assert len(_rendered_files(result)) == 3
        assert not [v for v in result.get_violations() if v.rule_code() == "TMP"]
    for linter in linters:
        linter.templater.close()
