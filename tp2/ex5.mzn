%-4p Quoridor tournament-

include "globals.mzn";

int: T = 14;%jours de tournoi
int: G = 2;%sites de jeu
int: P = 13;%nombre de joueurs

array[1..T,1..G,1..4] of var 0..P: x;

% <A COMPLETER>

solve satisfy;

output
["\n Site 1 \t Site 2\n"]
++
[
    "\n"++show(x[i,1,1])++" "++show(x[i,1,2])++" "++show(x[i,1,3])++" "++show(x[i,1,4])++" \t"++show(x[i,2,1])++" "++show(x[i,2,2])++" "++show(x[i,2,3])++" "++show(x[i,2,4])
  | i in 1..T
]