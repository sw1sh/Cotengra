(* Run the example dataset through the Cotengra Wolfram wrappers and print results *)
PacletDirectoryLoad["Cotengra"]
Needs["Wolfram`Cotengra`"]

inputs = {
  {3}, {1}, {4}, {2}, {5}, {7, 1, 2},
  {9, 6, 8, 3, 4, 5}, {10, 11, 6, 7, 8},
  {12, 13, 14, 16, 15, 9, 10, 11}, {17, 18, 12, 13, 14, 15},
  {24, 20, 16}, {19, 17}, {21, 22, 23, 25, 18, 19, 20},
  {31, 26, 29, 35, 30, 21, 22, 23, 24, 25}, {27, 26},
  {28, 27}, {33, 36, 28, 29, 30}, {32, 34, 31},
  {37, 38, 39, 32, 33, 34, 35, 36}, {40, 41, 42, 37}
}

output = {40, 41, 42, 38, 39}

sizes = <|
  3 -> 4, 1 -> 4, 4 -> 2, 2 -> 4, 5 -> 3, 7 -> 4, 9 -> 4, 6 -> 2, 8 -> 3,
  10 -> 4, 11 -> 3, 12 -> 4, 13 -> 4, 14 -> 2, 16 -> 4, 15 -> 3, 17 -> 4,
  18 -> 4, 24 -> 4, 20 -> 3, 19 -> 2, 21 -> 4, 22 -> 4, 23 -> 2, 25 -> 3,
  31 -> 4, 26 -> 4, 29 -> 2, 35 -> 4, 30 -> 3, 27 -> 4, 28 -> 4, 33 -> 4,
  36 -> 3, 32 -> 4, 34 -> 2, 37 -> 2, 38 -> 4, 39 -> 3, 40 -> 4, 41 -> 4,
  42 -> 2
|>

Print["Running GreedyPath..."]
gp = GreedyPath[inputs, output, sizes]
Print["GreedyPath result: " <> ToString[gp]]

Print["Running OptimalPath (minimize -> None)..."]
op = OptimalPath[inputs, output, sizes]
Print["OptimalPath result: " <> ToString[op]]

Print["Done."]
