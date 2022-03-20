%-2p Quoridor tournament-

include "globals.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;

% <A COMPLETER>

output
["Schedule ("++show(T)++" time slots)\n"]
++
[
    if j = 1 then "\n" else " " endif ++ show(x[i,j,1])++" "++show(x[i,j,2])++ if j = G then " " else " -" endif
      | i in 1..MAX_T, j in 1..G
]