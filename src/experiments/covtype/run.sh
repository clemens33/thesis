#!/bin/bash

# requires an env.sh file which sets all relevant environment variables
source ../env.sh

script="$1"


if [ "${script: -3}" == ".py" ]
then
  echo "run $script ..."

  python "$script"
else
  echo "run $script.py ..."

  python "$script.py"
fi
#nohup python "$script.py" > "$script.log" 2>&1 &
#echo "pid: $!" > "$script.pid"