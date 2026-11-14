"""Implements runner classes used internally by the Linter class.

Implements various runner types for SQLFluff:
- Serial
- Parallel
  - Multiprocess
  - Multithread (used only by automated tests)
"""

import bdb
import functools
import logging
import multiprocessing
import multiprocessing.dummy
import multiprocessing.pool
import os
import pickle
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Generator, Iterable, Iterator
from contextvars import ContextVar
from types import TracebackType
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Union

from sqlfluff.core import FluffConfig, Linter
from sqlfluff.core.errors import SQLFluffSkipFile, SQLFluffUserError, SQLTemplaterError
from sqlfluff.core.linter import LintedFile, RenderedFile
from sqlfluff.core.linter.common import DeferredRenderTask
from sqlfluff.core.linter.linter import _selectable_by_nested_config
from sqlfluff.core.plugin.host import is_main_process

if TYPE_CHECKING:  # pragma: no cover
    from sqlfluff.core.formatter import FormatterInterface
    from sqlfluff.core.templaters import RawTemplater

linter_logger: logging.Logger = logging.getLogger("sqlfluff.linter")

PartialLintCallable = Callable[[], LintedFile]


class _PreparedFile(NamedTuple):
    """A file, and the templater to render it."""

    fname: str
    templater: "RawTemplater"
    # Set if the templater failed to sequence its files, e.g. as its dbt
    # project doesn't compile. The file then reports that error.
    error: Optional[SQLFluffUserError] = None


class _WorkerState(NamedTuple):
    """What each parallel worker receives once, when its pool starts."""

    # The pickled root config, for templaters which render in the worker.
    pickled_root_config: Optional[bytes]
    # Whether the caller keeps each LintedFile. If not, workers can leave
    # out parse trees which the main process won't read.
    retain_files: bool
    # The root config, unpickled when the worker renders its first file.
    root_config: Optional[FluffConfig] = None


# Set in each worker by ParallelRunner._init_worker.
_worker_state: ContextVar[_WorkerState] = ContextVar(
    "_worker_state", default=_WorkerState(None, True)
)


class BaseRunner(ABC):
    """Base runner class."""

    def __init__(
        self,
        linter: Linter,
        config: FluffConfig,
    ) -> None:
        self.linter = linter
        self.config = config
        self.skipped_file_count: int = 0

    pass_formatter = True

    def _iter_prepared(
        self, fnames: list[str], formatter: Optional["FormatterInterface"]
    ) -> Iterator[_PreparedFile]:
        """Iterate through files with the templater for each, in render order.

        Files are grouped by templater, so each templater can sequence its
        own files. The files of a templater which renders in worker processes
        then fill the places in discovery order which they had. A templater
        which renders in the main process (e.g. dbt) may hold state for all
        its files, so its files are rendered together, from the place of its
        first file, and only one such templater holds its state at once.
        Each templater created for the operation is released after its last
        file.
        """
        with self.linter._templater_session() as session:
            groups: dict[int, list[tuple[int, _PreparedFile]]] = {}
            # The templater for the files in each directory, if it can be
            # chosen without their configs. Otherwise each file is loaded
            # here, to apply its inline config, and loaded again to render
            # it, so that its text isn't held meanwhile.
            templater_by_dir: dict[str, Optional["RawTemplater"]] = {}
            # The config of the first file of each group, if it was loaded.
            first_configs: dict[int, FluffConfig] = {}
            for index, fname in enumerate(fnames):
                dirname = os.path.dirname(fname)
                if dirname not in templater_by_dir:
                    templater_by_dir[dirname] = self.linter._templater_for_path(
                        session, fname
                    )
                templater = templater_by_dir[dirname]
                if templater is None:
                    try:
                        _, file_config, _ = self.linter.load_raw_file_and_config(
                            fname, self.config
                        )
                    except SQLFluffSkipFile as skip_file_err:
                        self._skip(skip_file_err)
                        continue
                    # Any warning is logged when the file is rendered.
                    templater = self.linter._templater_for_config(
                        session, file_config, fname, warn=False
                    )
                    first_configs.setdefault(id(templater), file_config)
                groups.setdefault(id(templater), []).append(
                    (index, _PreparedFile(fname, templater))
                )
            # The templater, and the files it renders, at each place in
            # discovery order.
            places: dict[int, tuple["RawTemplater", Iterator[_PreparedFile]]] = {}
            for group in groups.values():
                templater = group[0][1].templater
                sequenced = self._sequenced(
                    group,
                    formatter,
                    list(groups.values()),
                    first_configs.get(id(templater)),
                )
                if templater.templates_in_worker:
                    for (index, _), prepared in zip(group, sequenced):
                        places[index] = (templater, iter((prepared,)))
                else:
                    places[group[0][0]] = (templater, sequenced)
            last_places = {
                id(templater): index for index, (templater, _) in sorted(places.items())
            }
            for index, (templater, prepared_files) in sorted(places.items()):
                yield from prepared_files
                if last_places[id(templater)] == index:
                    session.release(templater)

    def _sequenced(
        self,
        group: list[tuple[int, _PreparedFile]],
        formatter: Optional["FormatterInterface"],
        groups: list[list[tuple[int, _PreparedFile]]],
        first_config: Optional[FluffConfig],
    ) -> Iterator[_PreparedFile]:
        """Iterate through the files of one templater, in its own order.

        Files are only sequenced once the first is needed, as a templater may
        load its state (e.g. a dbt manifest) to sequence them. If that fails
        with a user error, it's raised if all `groups` have the same scope,
        or else each file reports it.
        """
        first = group[0][1]
        templater = first.templater
        # The root templater is sequenced with the root config, and others
        # with the config of their first file: with its inline config if it
        # was loaded to choose the templater, or else from its path.
        sequence_config = self.config
        if templater is not self.linter.templater:
            sequence_config = first_config or self.config.make_child_from_path(
                first.fname, require_dialect=False
            )
        files_by_name: dict[str, deque[_PreparedFile]] = defaultdict(deque)
        for _, prepared in group:
            files_by_name[prepared.fname].append(prepared)
        try:
            fnames = list(
                templater.sequence_files(
                    [prepared.fname for _, prepared in group],
                    config=sequence_config,
                    formatter=formatter,
                )
            )
        except SQLFluffUserError as error:
            # This only stops the run if all files have the same templater
            # config, ignoring inline config. Otherwise the files of this
            # templater report the error, and the others are linted.
            if len({self._scope(other) for other in groups}) == 1:
                raise
            for _, prepared in group:
                yield prepared._replace(error=error)
            return
        for fname in fnames:
            yield files_by_name[fname].popleft()

    def _scope(self, group: list[tuple[int, _PreparedFile]]) -> tuple[str, ...]:
        """The key of a group's templater for its first file, without inline config.

        Groups which only differ by inline config have the same scope.
        """
        first = group[0][1]
        return first.templater.session_key(
            self.config.make_child_from_path(first.fname, require_dialect=False)
        )

    def _skip(self, skip_file_err: SQLFluffSkipFile) -> None:
        """Skip a file."""
        linter_logger.warning(str(skip_file_err))
        self.skipped_file_count += 1

    def _render(self, prepared: _PreparedFile) -> Optional[RenderedFile]:
        """Render a file, or return None if it's skipped."""
        try:
            raw_file, config, encoding = self.linter.load_raw_file_and_config(
                prepared.fname, self.config
            )
            if prepared.error:
                return RenderedFile(
                    [],
                    [SQLTemplaterError(str(prepared.error))],
                    config,
                    {"templating": 0.0},
                    prepared.fname,
                    encoding,
                    raw_file,
                )
            return self.linter.render_string(raw_file, prepared.fname, config, encoding)
        except SQLFluffSkipFile as skip_file_err:
            self._skip(skip_file_err)
            return None

    def _lint_partial(self, rendered: RenderedFile, fix: bool) -> PartialLintCallable:
        """Make a callable which lints a rendered file."""
        # Generate a fresh ruleset
        rule_pack = self.linter.get_rulepack(config=rendered.config)
        return functools.partial(
            self.linter.lint_rendered,
            rendered,
            rule_pack,
            fix,
            # Formatters may or may not be passed. They don't pickle
            # nicely so aren't appropriate in a multiprocessing world.
            self.linter.formatter if self.pass_formatter else None,
        )

    def iter_rendered(self, fnames: list[str]) -> Iterator[tuple[str, RenderedFile]]:
        """Iterate through rendered files ready for linting."""
        for prepared in self._iter_prepared(fnames, self.linter.formatter):
            rendered = self._render(prepared)
            if rendered is not None:
                yield prepared.fname, rendered

    def iter_partials(
        self,
        fnames: list[str],
        fix: bool = False,
    ) -> Iterator[tuple[str, Union[PartialLintCallable, DeferredRenderTask]]]:
        """Iterate through partials for linted files.

        Generates filenames and objects which return LintedFiles.
        """
        for fname, rendered in self.iter_rendered(fnames):
            yield fname, self._lint_partial(rendered, fix)

    @abstractmethod
    def run(self, fnames: list[str], fix: bool) -> Iterator[LintedFile]:
        """Run linting on the specified list of files."""
        ...

    @classmethod
    def _init_global(cls) -> None:
        """Initializes any global state.

        May be overridden by subclasses to apply global configuration, initialize
        logger state in child processes, etc.
        """
        pass

    @staticmethod
    def _handle_lint_path_exception(fname: Optional[str], e: BaseException) -> None:
        if isinstance(e, IOError):
            # IOErrors are caught in commands.py, so propagate it
            raise (e)  # pragma: no cover
        linter_logger.warning(
            f"""Unable to lint {fname} due to an internal error. \
Please report this as an issue with your query's contents and stacktrace below!
To hide this warning, add the failing file to .sqlfluffignore
{traceback.format_exc()}""",
        )


class SequentialRunner(BaseRunner):
    """Simple runner that does sequential processing."""

    def run(self, fnames: list[str], fix: bool) -> Iterator[LintedFile]:
        """Sequential implementation."""
        for fname, partial in self.iter_partials(fnames, fix=fix):
            try:
                if isinstance(partial, DeferredRenderTask):  # pragma: no cover
                    # DeferredRenderTask is normally only emitted by
                    # ParallelRunner.iter_partials.  Handle it here as a
                    # safety net: render + lint in one step in the main process.
                    rendered = self.linter.render_file(partial.fname, self.config)
                    rule_pack = self.linter.get_rulepack(config=rendered.config)
                    yield self.linter.lint_rendered(
                        rendered, rule_pack, partial.fix, self.linter.formatter
                    )
                else:
                    yield partial()
            except (bdb.BdbQuit, KeyboardInterrupt):  # pragma: no cover
                raise
            except Exception as e:
                self._handle_lint_path_exception(fname, e)


class ParallelRunner(BaseRunner):
    """Base class for parallel runner implementations (process or thread)."""

    POOL_TYPE: Callable[..., multiprocessing.pool.Pool]
    # Don't pass the formatter in a parallel world, they
    # don't pickle well.
    pass_formatter = False

    def __init__(
        self,
        linter: Linter,
        config: FluffConfig,
        processes: int,
        retain_files: bool = True,
    ) -> None:
        super().__init__(linter, config)
        self.processes = processes
        self.retain_files = retain_files

    def iter_partials(
        self,
        fnames: list[str],
        fix: bool = False,
    ) -> Generator[
        tuple[str, Union[PartialLintCallable, DeferredRenderTask]], None, None
    ]:
        """Iterate through partials or deferred tasks for parallel linting.

        For files whose templater supports worker-side rendering
        (``templates_in_worker = True``), we emit a lightweight
        ``DeferredRenderTask`` containing only the filename. The worker
        process calls ``render_file`` itself, with the root config it got
        when the pool started, keeping the full ``RenderedFile`` off the IPC
        boundary.

        For templaters that require main-process state (e.g. dbt), we
        template in the main process and emit a callable which lints the
        rendered file.
        """
        for prepared in self._iter_prepared(fnames, None):
            if prepared.templater.templates_in_worker:
                yield prepared.fname, DeferredRenderTask(prepared.fname, fix)
                continue
            rendered = self._render(prepared)
            if rendered is not None:
                yield prepared.fname, self._lint_partial(rendered, fix)

    def run(self, fnames: list[str], fix: bool) -> Iterator[LintedFile]:
        """Parallel implementation.

        Note that the partials are generated one at a time then
        passed directly into the pool as they're ready. This means
        the main thread can do the IO work while passing the parsing
        and linting work out to the threads.
        """
        # NOTE: We avoid using `with pool:` here because Pool.__exit__
        # calls pool.terminate() but NOT pool.join(). Without join(), worker
        # processes may still be alive when Python's resource_tracker runs at
        # shutdown, causing "leaked semaphore objects" warnings from the named
        # POSIX semaphores used by the pool's internal SimpleQueue locks.
        partials = self.iter_partials(fnames, fix=fix)
        pool = self._create_pool(
            self.processes,
            # Send each worker the root config once, rather than with every file.
            functools.partial(
                self._init_worker, self._root_config_for_workers(), self.retain_files
            ),
        )
        try:
            for lint_result in self._map(
                pool,
                self._apply,
                partials,
            ):
                if isinstance(lint_result, DelayedException):
                    if isinstance(lint_result.ee, SQLFluffSkipFile):
                        # A file was skipped (e.g. exceeded
                        # large_file_skip_byte_limit). Log a plain warning,
                        # not the "please report as bug" message.
                        linter_logger.warning(str(lint_result.ee))
                        self.skipped_file_count += 1
                    else:
                        try:
                            lint_result.reraise()
                        except Exception as e:
                            self._handle_lint_path_exception(lint_result.fname, e)
                else:
                    # It's a LintedDir.
                    if self.linter.formatter:
                        self.linter.formatter.dispatch_file_violations(
                            lint_result.path,
                            lint_result,
                            only_fixable=fix,
                            warn_unused_ignores=self.linter.config.get(
                                "warn_unused_ignores"
                            ),
                        )
                    yield lint_result
        except KeyboardInterrupt:  # pragma: no cover
            # On keyboard interrupt (Ctrl-C), terminate the workers.
            # Notify the user we've received the signal and are cleaning up,
            # in case it takes awhile.
            print("Received keyboard interrupt. Cleaning up and shutting down...")
        finally:
            try:
                pool.terminate()
            finally:
                pool.join()
                # Close the templater session now, even if we stopped early.
                partials.close()

    def _root_config_for_workers(self) -> Optional[bytes]:
        """The pickled root config to send each worker, so it can render files.

        It's needed if the root templater renders in workers, or if nested
        configuration may select one which does. Otherwise (e.g. sqlmesh) every
        file is rendered in the main process.
        """
        if not _selectable_by_nested_config(self.linter.templater):
            return None
        config = self.config.copy()
        # Workers build each file's config, with its own dialect, from the
        # file's path, so they never use the root config's expanded dialect.
        # Leave it out: it's most of a pickled config, and the main process
        # waits for each new worker to read a large start-up message before
        # starting the next one.
        config._configs["core"]["dialect_obj"] = None
        return pickle.dumps(config)

    @classmethod
    def _init_worker(
        cls, pickled_root_config: Optional[bytes], retain_files: bool
    ) -> None:
        """Set up a pool worker, and keep what it needs for every file."""
        cls._init_global()
        _worker_state.set(_WorkerState(pickled_root_config, retain_files))

    @staticmethod
    def _apply(
        partial_tuple: tuple[str, Union[PartialLintCallable, DeferredRenderTask]],
    ) -> Union["DelayedException", LintedFile]:
        """Shim function used in parallel mode."""
        fname, task = partial_tuple
        state = _worker_state.get()
        try:
            if isinstance(task, DeferredRenderTask):
                root_config = state.root_config
                if root_config is None:
                    # Unpickle the root config here, on the worker's first
                    # file, rather than in its initializer: an error is then
                    # reported for the file instead of killing the worker, and
                    # any plugins it loads see that this isn't the main process.
                    assert state.pickled_root_config, "Worker has no root config."
                    root_config = pickle.loads(state.pickled_root_config)
                    _worker_state.set(state._replace(root_config=root_config))
                # Worker-side rendering: reconstruct a Linter from the root
                # config and do render + lint in one step, keeping the full
                # RenderedFile off the IPC boundary.
                linter = Linter(config=root_config)
                # FluffConfig.__getstate__ strips templater_obj to None before
                # pickling (it's designed for main-process use only). Since we
                # are deliberately rendering here in the worker, re-instantiate
                # the templater from the config's templater name.
                linter.templater = root_config.get_templater()
                rendered = linter.render_file(task.fname, root_config)
                rule_pack = linter.get_rulepack(config=rendered.config)
                linted_file = Linter.lint_rendered(rendered, rule_pack, task.fix, None)
            else:
                linted_file = task()
            if state.retain_files:
                return linted_file
            return ParallelRunner._without_unused_tree(linted_file)
        # Capture any exceptions and return as delayed exception to handle
        # in the main thread.
        except Exception as e:
            return DelayedException(e, fname=fname)
        # A worker's SIGTERM handler may raise this during pool.terminate(). If we
        # kept the worker alive here, that call would wait for it forever.
        except SystemExit:
            raise
        # Any other exception, such as a Rust panic, would end this worker without
        # a result, and the main process would wait for it forever. It might not
        # pickle, so we send a description of it instead.
        except BaseException as e:
            # NOTE: This is safe to call even if the exception can't be printed.
            description = "".join(traceback.format_exception_only(type(e), e))
            return DelayedException(RuntimeError(description.strip()), fname=fname)

    @staticmethod
    def _without_unused_tree(linted_file: LintedFile) -> LintedFile:
        """Leave out a parse tree which the main process won't read.

        When files aren't retained, the main process only reads a file's
        tree to write its fixes and to count its segments. Large trees are
        slow to send between processes, so we keep the tree only when the
        file has fixable violations, and otherwise send the counts instead.
        """
        if not linted_file.tree or linted_file.num_violations(
            fixable=True, filter_warning=False
        ):
            return linted_file
        return linted_file._replace(
            tree=None,
            segment_counts=(
                linted_file.tree.count_segments(raw_only=False),
                linted_file.tree.count_segments(raw_only=True),
            ),
        )

    @classmethod
    def _init_global(cls) -> None:  # pragma: no cover
        """For the parallel runners indicate that we're not in the main thread."""
        is_main_process.set(False)
        super()._init_global()

    @classmethod
    def _create_pool(
        cls, processes: int, initializer: Callable[[], None]
    ) -> multiprocessing.pool.Pool:
        return cls.POOL_TYPE(processes=processes, initializer=initializer)

    @classmethod
    @abstractmethod
    def _map(
        cls,
        pool: multiprocessing.pool.Pool,
        func: Callable[
            [tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
            Union["DelayedException", LintedFile],
        ],
        iterable: Iterable[tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
    ) -> Iterable[Union["DelayedException", LintedFile]]:  # pragma: no cover
        """Class-specific map method.

        NOTE: Must be overridden by an implementation.
        """
        ...


class MultiProcessRunner(ParallelRunner):
    """Runner that does parallel processing using multiple processes."""

    # NOTE: Python 3.13 deprecates calling `Pool` without first setting
    # the context. The default was already "spawn" for MacOS and Windows
    # but was previously "fork" for other Linux platforms. From python
    # 3.14 onwards, the default will not be "fork" anymore.
    # In testing we've found no significant difference between "fork"
    # and "spawn", and so settle on "spawn" for all operating system.
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    POOL_TYPE = multiprocessing.get_context("spawn").Pool

    @classmethod
    def _init_global(cls) -> None:  # pragma: no cover
        super()._init_global()

        # Disable signal handling in the child processes to let the parent
        # control all KeyboardInterrupt handling (Control C). This is
        # necessary in order for keyboard interrupts to exit quickly and
        # cleanly. Adapted from this post:
        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    @classmethod
    def _map(
        cls,
        pool: multiprocessing.pool.Pool,
        func: Callable[
            [tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
            Union["DelayedException", LintedFile],
        ],
        iterable: Iterable[tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
    ) -> Iterable[Union["DelayedException", LintedFile]]:
        """Map using imap unordered.

        We use this so we can iterate through results as they arrive, and while other
        files are still being processed.
        """
        return pool.imap_unordered(func=func, iterable=iterable)


class MultiThreadRunner(ParallelRunner):
    """Runner that does parallel processing using multiple threads.

    Used only by automated tests.
    """

    POOL_TYPE = multiprocessing.dummy.Pool

    @classmethod
    def _map(
        cls,
        pool: multiprocessing.pool.Pool,
        func: Callable[
            [tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
            Union["DelayedException", LintedFile],
        ],
        iterable: Iterable[tuple[str, Union[PartialLintCallable, DeferredRenderTask]]],
    ) -> Iterable[Union["DelayedException", LintedFile]]:
        """Map using imap.

        We use this so we can iterate through results as they arrive, and while other
        files are still being processed.
        """
        return pool.imap(func=func, iterable=iterable)


class DelayedException(Exception):
    """Multiprocessing process pool uses this to propagate exceptions."""

    def __init__(self, ee: BaseException, fname: Optional[str] = None):
        self.ee = ee
        self.tb: Optional[TracebackType]
        _, _, self.tb = sys.exc_info()
        self.fname = fname
        super().__init__(str(ee))

    def reraise(self) -> None:
        """Reraise the encapsulated exception."""
        raise self.ee.with_traceback(self.tb)


def get_runner(
    linter: Linter,
    config: FluffConfig,
    processes: int,
    allow_process_parallelism: bool = True,
    retain_files: bool = True,
) -> tuple[BaseRunner, int]:
    """Generate a runner instance based on parallel and system configuration.

    The processes argument can be positive or negative.
    - If positive, the integer is interpreted as the number of processes.
    - If negative or zero, the integer is interpreted as number_of_cpus - processes.

    e.g.
    -1 = all cpus but one.
    0 = all cpus
    1 = 1 cpu

    The retain_files argument tells parallel runners whether the caller keeps
    each LintedFile (see `Linter.lint_paths`).
    """
    if processes <= 0:
        processes = max(multiprocessing.cpu_count() + processes, 1)

    if processes > 1:
        # Process parallelism isn't really supported during testing
        # so this flag allows us to fall back to a threaded runner
        # in those cases.
        if allow_process_parallelism:
            return (
                MultiProcessRunner(
                    linter, config, processes=processes, retain_files=retain_files
                ),
                processes,
            )
        else:
            return (
                MultiThreadRunner(
                    linter, config, processes=processes, retain_files=retain_files
                ),
                processes,
            )
    else:
        return SequentialRunner(linter, config), processes
