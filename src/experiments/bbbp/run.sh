#!/bin/bash

export PYTHONPATH="$PYTHONPATH:../../../" # add root to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:../../"    # add src to PYTHONPATH
export PYTHONUNBUFFERED=1

script="$1"

echo "run $script.py ..."

python "$script.py"

#nohup python "$script.py" > "$script.log" 2>&1 &
#echo "pid: $!" > "$script.pid"
