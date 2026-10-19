#!/usr/bin/env bash
# Run from WSL, in the repo, after `terraform apply` in infra/gpu (docs/gpu-runbook.md).
#
#     scripts/gpu/local.sh push    <ip>             # copy the committed repo (it is private: no git clone)
#     scripts/gpu/local.sh setup   <ip>             # start setup.sh on the instance, detached
#     scripts/gpu/local.sh wait-setup <ip>          # wait until setup.sh finished (SETUP_DONE / SETUP_FAILED)
#     scripts/gpu/local.sh cells   <ip> <list>      # start run_cells.py with a cell list, detached
#     scripts/gpu/local.sh collect <ip> [dest]      # copy ~/out back every minute until DONE
#     scripts/gpu/local.sh status  <ip>             # setup / cell progress at a glance
#
# setup and cells run under setsid + nohup, so a dropped SSH connection does not
# stop them; collect is the only thing that has to keep running here, and the
# instance's data is lost at the auto-shutdown unless it has been copied.
#
# The launching ssh must return at once (docs/gpu-smoke-2026-10-18.md D5): the
# `&` covers only the detached command, never an `&&` chain (a chain would run
# in a subshell that holds ssh's output open until the job ends), and ssh -n
# closes its input. Whether a job finished is read from the marker files the
# job writes (wait-setup, collect, status), never from ssh returning.
set -euo pipefail

KEY=~/.ssh/id_ed25519
cmd=${1:?push|setup|cells|collect|status}
ip=${2:?the instance IP (terraform output gpu_public_ip)}
remote=ubuntu@$ip
ssh_() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$remote" "$@"; }
detach_() { ssh -n -i "$KEY" -o StrictHostKeyChecking=accept-new "$remote" "$@"; }

case "$cmd" in
  push)
    # The committed tree only: uncommitted edits would make a run that no commit reproduces.
    if ! git diff --quiet HEAD; then
      echo "uncommitted changes; commit first so the run matches a commit" >&2; exit 1
    fi
    ssh_ "rm -rf ~/toolcall-serving-bench && mkdir -p ~/toolcall-serving-bench ~/out"
    git archive HEAD | ssh_ "tar -x -C ~/toolcall-serving-bench"
    git rev-parse HEAD | ssh_ "cat > ~/out/repo_commit.txt"
    echo "pushed $(git rev-parse --short HEAD)"
    ;;
  setup)
    detach_ "mkdir -p ~/out; setsid nohup bash ~/toolcall-serving-bench/scripts/gpu/setup.sh > ~/out/setup.log 2>&1 < /dev/null &"
    echo "setup started; wait for it with: scripts/gpu/local.sh wait-setup $ip"
    ;;
  wait-setup)
    # collect does not watch SETUP_DONE, so a finished setup had no signal of its own.
    misses=0
    while true; do
      if state=$(ssh_ -o ConnectTimeout=20 'cd ~/out && for f in SETUP_DONE SETUP_FAILED; do test -f $f && echo $f; done; true'); then
        misses=0
        case "$state" in
          *SETUP_FAILED*)
            echo "setup.sh failed:" >&2; ssh_ 'cat ~/out/SETUP_FAILED; tail -n 20 ~/out/setup.log' >&2; exit 1 ;;
          *SETUP_DONE*)
            echo "setup done ($(date '+%H:%M:%S'))"; break ;;
        esac
      else
        misses=$((misses + 1))
        echo "instance unreachable ($misses in a row)" >&2
        if [ "$misses" -ge 10 ]; then echo "unreachable for 10 minutes; stopped" >&2; exit 1; fi
      fi
      sleep 30
    done
    ;;
  cells)
    list=${3:?a cell list, e.g. scripts/gpu/cells/smoke.json}
    ssh_ "test -f ~/out/SETUP_DONE" || { echo "setup has not finished" >&2; exit 1; }
    detach_ "rm -f ~/out/DONE; setsid nohup ~/venv/bin/python ~/toolcall-serving-bench/scripts/gpu/run_cells.py ~/toolcall-serving-bench/$list > ~/out/cells-$(basename "$list" .json).log 2>&1 < /dev/null &"
    echo "cells started: $list"
    ;;
  collect)
    dest=${3:-runs/gpu-$(date +%Y%m%d)}
    mkdir -p "$dest"
    misses=0
    while true; do
      # The markers are read on the instance: rsync -a never deletes, so a local
      # copy of RUNNING would outlive the run (and --delete could wipe an earlier
      # apply's results collected into the same directory).
      if rsync -a -e "ssh -i $KEY -o ConnectTimeout=20" "$remote:out/" "$dest/" &&
         state=$(ssh_ -o ConnectTimeout=20 'cd ~/out && for f in RUNNING DONE FAILED SETUP_FAILED; do test -f $f && echo $f; done; true'); then
        misses=0
      else
        misses=$((misses + 1))
        echo "instance unreachable ($misses in a row)" >&2
        # 10 minutes: the instance is gone (auto-shutdown) or this line's IP
        # changed (docs/gpu-runbook.md, section 4). Stop rather than loop forever.
        if [ "$misses" -ge 10 ]; then
          echo "unreachable for 10 minutes; stopped. What was copied is in $dest" >&2
          exit 1
        fi
        sleep 60
        continue
      fi
      case "$state" in
        *RUNNING*) ;;
        *SETUP_FAILED*)
          echo "setup.sh failed -- see $dest/SETUP_FAILED and $dest/setup.log" >&2
          exit 1 ;;
        *FAILED*)
          echo "the cell list stopped early -- see $dest/FAILED:" >&2
          cat "$dest/FAILED" >&2
          exit 1 ;;
        *DONE*)
          echo "collected into $dest ($(head -n 1 "$dest/DONE"))"
          break ;;
      esac
      sleep 60
    done
    ;;
  status)
    ssh_ 'for f in SETUP_DONE SETUP_FAILED RUNNING DONE FAILED; do test -f ~/out/$f && echo "$f: $(cat ~/out/$f 2>/dev/null)"; done; echo "--- status.jsonl"; cat ~/out/status.jsonl 2>/dev/null; echo "--- setup.log (tail)"; tail -n 3 ~/out/setup.log 2>/dev/null; echo "--- shutdown"; shutdown --show 2>/dev/null || cat /run/systemd/shutdown/scheduled 2>/dev/null'
    ;;
  *)
    echo "unknown command: $cmd" >&2; exit 1
    ;;
esac
