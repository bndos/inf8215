%-4p Quoridor tournament-

include "globals.mzn";

int: T = 14;%jours de tournoi
int: G = 2;%sites de jeu
int: P = 13;%nombre de joueurs

array[1..T,1..G,1..4] of var 0..P: x;

% <A COMPLETER>
% contrainte pour supprimer les matchs doublons
predicate all_different_exclude_zero(array[int] of var int: a) =
  forall (i, j in index_set(a) where i<j) (
    a[i] != 0 /\ a[j] !=0 -> a[i]!=a[j]
  );


constraint forall(t in 1..T, g in 1..G) (
           x[t,g,1] < x[t,g,2] /\ x[t,g,2] < x[t,g,3] /\ x[t,g,3] < x[t,g,4] /\ x[t,g,1] > 0 \/
           x[t,g,1] = 0 /\ x[t,g,2] = 0 /\ x[t,g,3] = 0 /\ x[t,g,4] = 0
);

constraint forall(t in 1..T, g in 1..G ) (
           sum([1 | j in 1..4 where x[t,g,j] = 0]) = 0 \/ sum([1 | j in 1..4 where x[t,g,j] = 0]) = 4
);

constraint all_different_exclude_zero([x[t, g, 1]+100*x[t,g,2]+2000*x[t,g,3]+40000*x[t,g,4] | g in 1..G, t in 1..T]);


%Chaque joueur doit avoir fait entre 8 et 11 matchs
array[int] of var int: l = [sum([1 | g in 1..G, t in 1..T, j in 1..4 where x[t,g,j] = p]) | p in 1..P];
constraint forall(a in l) (
      a >= 8 /\ a <=11
);



% Chaque joueur doit avour joué au moins une fois sur chaque site
constraint forall(p in 1..P) (
      sum([1 | t in 1..T, j in 1..4 where x[t,1,j] = p]) >= 1 /\ sum([1 | t in 1..T, j in 1..4 where x[t,2,j] = p]) >= 1
);


% Un joueur ne peut pas jouer contre lui meme
constraint forall(p in 1..P) (
      forall(t in 1..T) (
        sum([1 | g in 1..G where x[t,g,1] = p \/ x[t,g,2] = p \/ x[t,g,3] = p \/ x[t,g,4] = p]) <= 1
      )
);

% Chaque joueur doit avoir rencontre chaque autre joueur en match
constraint forall(p1 in 1..P) (
      forall(p2 in 1..P where p1<p2) (
        sum([1 | t in 1..T, g in 1..G, a,b in index_set(x[t,g,..]) where x[t, g, a] = p1 /\ x[t,g,b] = p2]) > 0
      )
);

% Contrainte de repos
constraint forall(p in 1..P) (
       forall(t in 1..(T-1)) (
          forall(j in 1..4)
          (
            forall(k in 1..4) (
              not ((x[t,1,j] = p /\ x[t+1,2,k] = p) \/ (x[t,2,j] = p /\ x[t+1,1,k] = p))
          )
          )
        % sum([1 | j in 1..4 where (x[t,1,j] = p /\ x[t+1,2,j] = p) \/ (x[t,2,j] = p /\ x[t+1,1,j] = p)]) = 0
       )
);


solve satisfy;

output
["\n Site 1 \t Site 2\n"]
++
[
    "\n"++show(x[i,1,1])++" "++show(x[i,1,2])++" "++show(x[i,1,3])++" "++show(x[i,1,4])++" \t"++show(x[i,2,1])++" "++show(x[i,2,2])++" "++show(x[i,2,3])++" "++show(x[i,2,4])
  | i in 1..T
]
