"""scripts/run_all_experiments.sh, run against a fake ``docker`` that records its calls."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_all_experiments.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not SCRIPT.exists(), reason="needs bash")

FAKE_DOCKER = r"""#!/usr/bin/env bash
# Records every call; imitates what the containers would write into ./results.
echo "$*" >> "$FAKE_LOG"
args="$*"
if [[ -n "${FAKE_FAIL:-}" && "$args" == *"$FAKE_FAIL"* ]]; then exit 1; fi
if [[ "$args" == "compose version" ]]; then exit 0; fi
if [[ "$args" == *"--seed"* && "$args" == *"--out /results/"* ]]; then
  run="${args##*--out /results/}"; run="${run%% *}"
  mkdir -p "$FAKE_ROOT/results/$run"
  echo '{"completed": true}' > "$FAKE_ROOT/results/$run/experiment.json"
fi
if [[ "$args" == *"detector.cli evaluate"* ]]; then
  out="${args##*--out /results/}"; out="${out%% *}"
  mkdir -p "$FAKE_ROOT/results/$out"
  echo '{}' > "$FAKE_ROOT/results/$out/results.json"
fi
exit 0
"""


@pytest.fixture()
def project(tmp_path):
    """A copy of the script in a fresh project folder, with the fake docker first on PATH."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "results").mkdir()
    return tmp_path


def run_script(project: Path, *args, **env):
    environ = {**os.environ, "PATH": f"{project / 'bin'}:{os.environ['PATH']}",
               "FAKE_LOG": str(project / "docker.log"), "FAKE_ROOT": str(project), "FLUSH_WAIT_S": "0", **env}
    proc = subprocess.run(["bash", str(project / "scripts" / SCRIPT.name), *args], env=environ,
                          capture_output=True, text=True, timeout=120)
    calls = (project / "docker.log").read_text().splitlines() if (project / "docker.log").exists() else []
    return proc, calls


def experiment_calls(calls):
    return [c for c in calls if "--seed" in c]


def test_full_run_order_and_outputs(project):
    proc, calls = run_script(project)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Expected total run time: about 15 h 50 min" in proc.stdout
    assert [c.split("--seed ")[1].split()[0] for c in experiment_calls(calls)] == ["1", "2", "3", "4", "5"]
    assert "up -d --build --wait" in calls[1]
    # per seed: start, clear logs, experiment, copy logs, evaluate
    i_clear = next(i for i, c in enumerate(calls) if "truncate -s 0" in c)
    i_exp = next(i for i, c in enumerate(calls) if "--seed 1 " in c)
    i_copy = next(i for i, c in enumerate(calls) if "cp -a /logs/. /results/run_1/logs/" in c)
    i_eval = next(i for i, c in enumerate(calls) if "evaluate" in c and "run_1/" in c)
    assert i_clear < i_exp < i_copy < i_eval
    assert "--duration-min 180 --warmup-min 10 --rps 5 --seed 1 --out /results/run_1" in calls[i_exp]
    assert calls[-2].endswith("aggregate --runs /results/run_1 /results/run_2 /results/run_3 /results/run_4 "
                              "/results/run_5 --out /results/summary")
    assert calls[-1].endswith("cross --train-run /results/run_1 --test-run /results/run_2 --out /results/cross_1_2")
    assert not (project / ".run-all-experiments.lock").exists()          # removed at exit


def test_completed_runs_are_skipped_on_restart(project):
    done = project / "results" / "run_1"
    (done / "detector").mkdir(parents=True)
    (done / "experiment.json").write_text(json.dumps({"completed": True}))
    (done / "detector" / "results.json").write_text("{}")
    proc, calls = run_script(project)
    assert proc.returncode == 0, proc.stderr
    assert "[run_1] already complete and evaluated - skipped" in proc.stdout
    assert not any("--seed 1 " in c for c in calls)
    assert any("--seed 2 " in c for c in calls)
    assert "/results/run_1 /results/run_2" in calls[-2]                   # still aggregated


def test_half_finished_run_stops_the_script(project):
    (project / "results" / "run_2").mkdir()
    (project / "results" / "run_2" / "experiment.json").write_text(json.dumps({"completed": False}))
    proc, calls = run_script(project)
    assert proc.returncode == 4
    assert "run_2 exists but is not a complete, evaluated run" in proc.stderr
    assert [c.split("--seed ")[1].split()[0] for c in experiment_calls(calls)] == ["1"]


def test_refuses_while_another_experiment_runs(project):
    (project / "results" / ".experiment-running").write_text('{"out": "/results/manual"}')
    proc, calls = run_script(project)
    assert proc.returncode == 3
    assert "another experiment is running" in proc.stderr
    assert not any("up -d" in c or "truncate" in c for c in calls)


def test_refuses_a_second_copy_of_itself(project):
    (project / ".run-all-experiments.lock").write_text("12345")
    proc, calls = run_script(project)
    assert proc.returncode == 3 and "already running" in proc.stderr
    assert not any("up -d" in c for c in calls)
    assert (project / ".run-all-experiments.lock").read_text() == "12345"   # not removed by the refused copy


def test_stops_at_the_first_failing_step(project):
    proc, calls = run_script(project, FAKE_FAIL="detector.cli evaluate --logs /results/run_1/logs")
    assert proc.returncode != 0
    assert "step 'run_1: evaluate' failed" in proc.stdout
    assert not any("--seed 2 " in c for c in calls)
    assert not any("aggregate" in c for c in calls)
    assert not (project / ".run-all-experiments.lock").exists()


def test_dry_run_and_settings(project):
    proc, calls = run_script(project, "--dry-run", SEEDS="7 8", DURATION_MIN="30")
    assert proc.returncode == 0, proc.stderr
    assert calls == []                                                     # docker never called
    assert "+ docker compose --profile experiment run --rm experiment --duration-min 30" in proc.stdout
    assert "--out /results/cross_7_8" in proc.stdout
    assert "about 1 h 26 min" in proc.stdout                               # 2 x (30 + 8) + 10
    proc, _ = run_script(project, "--dry-run", SEEDS="1")
    assert proc.returncode == 2 and "at least two seeds" in proc.stderr
    proc, _ = run_script(project, "--bogus")
    assert proc.returncode == 2
