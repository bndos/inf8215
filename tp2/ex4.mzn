%-2p Quoridor tournament-

include "globals.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs
int: N = P * (P-1) div 2;%nombre de matchs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;

% <A COMPLETER>

% Joueur 1 est deja assigne a des creneaux
array[int] of int: player1 = ([p | p in 1..P + P div 3 where (p mod 3) != 0 ]);
constraint forall(p in 1..P-1) (
    x[player1[p], 1, 1] = 1 /\
    x[player1[p], 1, 2] = p + 1
);


% Les jouers ne peuvent pas jouer sur plus d'une table en meme temps
constraint forall(t in 1..MAX_T, p in 1..P) (
      sum([1 | g in 1..G where x[t, g, 1] = p \/ x[t, g, 2] = p]) <= 1
);

% On ne repete pas les parties
constraint alldifferent([x[t, g, 1] + x[t, g, 2] * x[t, g, 2] | g in 1..G,
                                                                t in 1..MAX_T
                                                                where x[t, g, 1] != 0]);

% Le premier joueur est plus petit que le deuxieme pour reduire les possibilites
% Aucun joueur ne joue contre 0
constraint forall(t in 1..T, g in 1..G)(
                  (0 < x[t, g, 1] /\ x[t, g, 1] < x[t, g, 2]) \/
                  (x[t, g, 1] = 0 /\ x[t, g, 2] = 0)
);

% Les joueurs ne jouent pas 3 matchs daffile
constraint forall(t in 1..MAX_T - 2, p in 1..P) (
           sum(g in 1..G where x[t, g, 1] = p \/ x[t, g, 2] = p)(1) +
           sum(g in 1..G where x[t+1, g, 1] = p \/ x[t+1, g, 2] = p)(1) +
           sum(g in 1..G where x[t+2, g, 1] = p \/ x[t+2, g, 2] = p)(1) <= 2
);

% On doit faire N match pour que tous les joueurs joue l'un contre l'autre
constraint sum(g in 1..G, t in 1..T where x[t, g, 1] != 0) (1) = N;

% On remplis le reste des lignes de 0
constraint forall(g in 1..G, t in T + 1..MAX_T) (x[t, g, 1] = 0 /\ x[t, g, 2] = 0);

solve minimize T;

output
["Schedule ("++show(T)++" time slots)\n"]
++
[
    if j = 1 then "\n" else " " endif ++ show(x[i,j,1])++" "++show(x[i,j,2])++ if j = G then " " else " -" endif
      | i in 1..MAX_T, j in 1..G
]
