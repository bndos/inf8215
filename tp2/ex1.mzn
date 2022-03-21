%-Carres-

include "globals.mzn";

array[1..5] of var 10..100: x;

% constraint <A COMPLETER>;
constraint sum(i in 1..4) (x[i]^2) = x[5]^2;

% solve <A COMPLETER>;
solve minimize x[5]^2;

output [
  show(x)
];

% Answer

% [10, 10, 10, 10, 20]