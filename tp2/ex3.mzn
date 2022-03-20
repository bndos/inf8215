%-Attribution de taches-

include "globals.mzn";

array[1..N,1..S] of int: cost;%cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration;%duree d'une tache dependamment du sous-traitant, en jours

int: S;%nombre de sous-traitants
int: N;%nombre de tâches
int: T;%horizon temporel, en jours

% Sous-traitants assigne aux taches
array[1..N] of var 1..S: x;

var int: C = sum(i in 1..N)(cost[i, x[i]]);

% <A COMPLETER>

% Les taches 1 et 2 sont exclusives
constraint forall(i in 2..N)( x[i] != x[1]);
constraint forall(i in 3..N)( x[i] != x[2]);

% La duree totale de tous les sous-traitants ne doit pas depasser T
constraint forall(s in 1..S)(
                 sum(i in 1..N)( % Plus rapide que (where x[i] == s)
                      if x[i] == s then
                          duration[i,s]
                      else 0 endif
                 ) <= T
);


solve minimize C;

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C)]
;
