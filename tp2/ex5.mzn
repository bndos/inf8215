
%-4p Quoridor tournament-

include "globals.mzn";

int: T = 14; % jours de tournoi
int: G = 2; % sites de jeu
int: P = 13; % nombre de joueurs

array[1..T,1..G,1..4] of var int: x;

array[1..T, 1..P] of var 0..G: sites;
array[1..P] of var int: played;
array[1..P, 1..P] of var int: opponents;

% condition: 4 joueurs par partie
constraint forall(t in 1..T, g in 1..G)(
    if g in sites[t, 1..P]then
        count(sites[t, 1..P], g) == 4
    endif
);

% condition: un joueur doit affronter chacun des autres joueurs
constraint forall(p in 1..P, p2 in 1..P)(
    opponents[p,p2] = sum(t in 1..T where sites[t, p] == sites[t, p2] /\ sites[t, p] != 0
)(1));

% condition: on ne peut pas jouer contre 0
constraint forall(i in 1..P)(count(opponents[i, 1..P], 0) == 0);

% condition: chaque joueur doit faire entre 8 et 11 matchs
constraint forall(i in 1..P)(played[i] = count(sites[1..T, i], 1) + count(sites[1..T, i], 1));
constraint forall(i in 1..P)(played[i] >= 8  /\ played[i] <= 11);

% condition: chaque joueur doit jouer au moins une fois sur chaque site
constraint forall(i in 1..P, j in 1..G)(j in sites[1..T, i]);

% condition: un joueur peut seulement jouer un des 2 matchs par jour s'ils ne sont pas dans la meme ville
constraint forall(p in 1..P, t in 1..(T - 1))(
  if sites[t, p] != 0 then
     sites[p, t] == sites[t + 1, p] \/ sites[t + 1, p] == 0 endif
);

solve satisfy;

output
["\n Site 1 \t\t Site 2\n"]
++
[
  "\n"++if length([k | k in 1..P where sites[i, k] == 1]) != 0 then
            show([k | k in 1..P where sites[i, k] == 1])
        else "[0, 0, 0, 0]" endif++"\t\t"++
        if length([k | k in 1..P where sites[i, k] == 2]) != 0 then
            show([k | k in 1..P where sites[i, k] == 2])
        else "[0, 0, 0, 0]" endif
  | i in 1..T
]
