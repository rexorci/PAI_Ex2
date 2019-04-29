#!/bin/bash
echo "Start Tournament"
source ../../venv/bin/activate

script1=chriweb_a2
script2=chriweb_a1
player1=IntrepidIbex
player2=CSA

n_maps=20

ts=$(date "+%Y%m%d%H%M%S")


log_name=$(echo tournament/${ts}/log/tournament_result.log)
log_name_rev=$(echo tournament/${ts}/log/tournament_result_rev.log)

mkdir tournament/${ts}
mkdir tournament/${ts}/log
mkdir tournament/${ts}/maps

for i in $(seq -w 001 $n_maps);
do
   python map_generator.py -n "tournament/${ts}/maps/tournament_${i}"
done

for i in $(seq -w 001 $n_maps);
do
    python kingsheep_tournament.py tournament/${ts}/maps/tournament_${i}.map -p1m $script1 -p1n $player1 -p2m $script2 -p2n $player2 >> "$log_name" &
    python kingsheep_tournament.py tournament/${ts}/maps/tournament_${i}.map -p1m $script2 -p1n $player2 -p2m $script1 -p2n $player1 >> "$log_name_rev" &
done

deactivate

echo "End Tournament"