#!/usr/bin/env bash
# Run from WSL, in the repo, after `terraform apply` in infra/gpu (docs/gpu-runbook.md).
#
#     scripts/gpu/local.sh push    <ip>             # copy the committed repo (it is private: no git clone)
#     scripts/gpu/local.sh setup   <ip>             # start setup.sh on the instance, detached
#     scripts/gpu/local.sh cells   <ip> <list>      # start run_cells.py with a cell list, detached
#     scripts/gpu/local.sh collect <ip> [dest]      # copy ~/out back every minute until DONE
#     scripts/gpu/local.sh status  <ip>             # setup / cell progress at a glance
#
# setup and cells run under nohup, so a dropped SSH connection does not stop
# them; collect is the only thing that has to keep running here, and the
# instance's data is lost at the auto-shutdown unless it has been copied.
set -euo pipefail

KEY=~/.ssh/id_ed25519
cmd=${1:?push|setup|cells|collect|status}
ip=${2:?the instance IP (terraform output gpu_public_ip)}
remote=ubuntu@$ip
ssh_() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$remote" "$@"; }

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
    ssh_ "mkdir -p ~/out && nohup bash ~/toolcall-serving-bench/scripts/gpu/setup.sh > ~/out/setup.log 2>&1 < /dev/null &"
    echo "setup started; follow it with: ssh -i $KEY $remote tail -f ~/out/setup.log"
    ;;
  cells)
    list=${3:?a cell list, e.g. scripts/gpu/cells/smoke.json}
    ssh_ "test -f ~/out/SETUP_DONE" || { echo "setup has not finished" >&2; exit 1; }
    ssh_ "rm -f ~/out/DONE && cd ~/toolcall-serving-bench && nohup ~/venv/bin/python scripts/gpu/run_cells.py $list > ~/out/cells-$(basename "$list" .json).log 2>&1 < /dev/null &"
    echo "cells started: $list"
    ;;
  collect)
    dest=${3:-runs/gpu-$(date +%Y%m%d)}
    mkdir -p "$dest"
    while true; do
      rsync -a -e "ssh -i $KEY" "$remote:out/" "$dest/" || echo "rsync failed; retrying" >&2
      if ssh_ "test -f ~/out/DONE && ! test -f ~/out/RUNNING" 2>/dev/null; then
        rsync -a -e "ssh -i $KEY" "$remote:out/" "$dest/"
        echo "collected into $dest ($(ssh_ cat ~/out/DONE))"
        break
      fi
      sleep 60
    done
    ;;
  status)
    ssh_ 'for f in SETUP_DONE RUNNING DONE; do test -f ~/out/$f && echo "$f: $(cat ~/out/$f 2>/dev/null)"; done; echo "--- status.jsonl"; cat ~/out/status.jsonl 2>/dev/null; echo "--- setup.log (tail)"; tail -n 3 ~/out/setup.log 2>/dev/null; shutdown --show 2>&1 | tail -n 1'
    ;;
  *)
    echo "unknown command: $cmd" >&2; exit 1
    ;;
esac
