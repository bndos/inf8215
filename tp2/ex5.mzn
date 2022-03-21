
%-4p Quoridor tournament-

include "globals.mzn";

int: T = 14; % jours de tournoi
int: G = 2; % sites de jeu
int: P = 13; % nombre de joueurs

array[1..T,1..G,1..4] of var int: x;

array[1..T, 1..P] of var 0..G: sites;
array[1..P] of var int: played;
array[1..P, 1..P] of var int: opponents;

% s'il y a une partie, 4 joueurs doivent jouer
constraint forall(t in 1..T, g in 1..G)(
    if g in sites[t, 1..P]then
        count(sites[t, 1..P], g) == 4
    endif
);

% matchs [1, 2, 3, 4] et [1, 2, 4, 3] sont équivalent
% // assurer par la représentation //

% chaque joueur doit avoir rencontré chaque autre joueur en match
constraint forall(p in 1..P, p2 in 1..P)(
    opponents[p,p2] = sum(t in 1..T where sites[t, p] == sites[t, p2] /\ sites[t, p] != 0
)(1));

constraint forall(i in 1..P)(count(opponents[i, 1..P], 0) == 0);

% chaque joueur doit avoir fait entre 8 et 11 matchs
constraint forall(i in 1..P)(played[i] = count(sites[1..T, i], 1) + count(sites[1..T, i], 1));
constraint forall(i in 1..P)(8 <= played[i]  /\ played[i] <= 11);

% chaque joueur doit avoir joué au moins une fois sur chaque site
constraint forall(i in 1..P, j in 1..G)(j in sites[1..T, i]);

% un joueur ne peut jouer qu’un seul des 2 matchs par jour
% // assurer par la représentation //

% au moins une journée de repos entre deux matchs si ceux-ci ne sont pas dans la même ville
constraint forall(i in 1..P, j in 1..(T - 1))(
  if sites[j, i] != 0 then
     sites[i, j] == sites[j + 1, i] \/ sites[j + 1, i] == 0 endif
);

solve satisfy;

output
["\n Site 1 \t\t Site 2\n"]
++
[
  "\n"++if length([k | k in 1..P where sites[i, k] == 1]) != 0 then show([k | k in 1..P where sites[i, k] == 1]) else "[0, 0, 0, 0]" endif++"\t\t"++
  if length([k | k in 1..P where sites[i, k] == 2]) != 0 then show([k | k in 1..P where sites[i, k] == 2]) else "[0, 0, 0, 0]" endif
  | i in 1..T
]

% -Réponse-
%   Site 1            Site 2

%[1,  3,  5,  6 ]  [2,  7,  9,  12]
%[3,  4,  8,  11]  [2,  7,  10, 13]
%[1,  3,  5,  12]  [6,  9,  10, 13]
%[1,  2,  5,  7 ]  [6,  9,  10, 11]
%[2,  3,  5,  7 ]  [4,  8,  10, 13]
%[1,  5,  9,  11]  [4,  6,  8,  12]
%[1,  3,  10, 11]  [4,  6,  8,  12]
%[1,  7,  11, 13]  [2,  4,  6,  8 ]
%[5,  7,  10, 13]  [3,  4,  8,  9 ]
%[2,  7,  11, 13]  [4,  6,  8,  12]
%[1,  5,  9,  11]  [4,  6,  8,  12]
%[2,  3,  9,  10]  [4,  6,  8,  12]
%[2,  3,  9,  13]  [4,  6,  7,  8 ]
%[10, 11, 12, 13]  [1,  4,  5,  8 ]
% Site 1 		 Site 2
%
%[4, 5, 7, 8]    [2, 9, 11, 12]
%[3, 4, 6, 9]    [1, 2, 10, 12]
%[4, 10, 11, 13] [2, 3, 5, 8]
%[1, 2, 7, 9]    [3, 4, 12, 13]
%[7, 8, 10, 12]  [4, 6, 9, 11]
%[1, 2, 3, 11]   [5, 6, 7, 10]
%[4, 10, 12, 13] [5, 6, 9, 11]
%[2, 8, 9, 13]   [1, 3, 6, 12]
%[1, 7, 8, 13]   [3, 4, 5, 10]
%[2, 6, 11, 12]  [5, 7, 9, 10]
%[3, 5, 12, 13]  [1, 6, 8, 10]
%[4, 6, 10, 13]  [1, 2, 5, 7]
%[7, 8, 9, 11]   [1, 2, 3, 4]
%[3, 7, 8, 11]   [4, 5, 6, 13]
