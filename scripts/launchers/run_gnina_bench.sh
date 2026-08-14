#!/bin/bash
# Full Gnina cross-docking of the PLINDER panel. In a script file, not an inline command, so
# that `pgrep -f gnina_benchmark` cannot match the shell that launched it (CLAUDE.md S6).
cd /home/judewells/plixer_outer/plixer
exec ./venvPlixer/bin/python scripts/adhoc_analysis/gnina_benchmark.py --per_gpu 4
