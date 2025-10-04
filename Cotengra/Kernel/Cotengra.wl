BeginPackage["Wolfram`Cotengra`"];

GreedyPath
OptimalPath
ContractIndices
TreePathToPath
PathToTreePath
CanonicalPath
PathIndexContractions


Begin["`Private`"];

$libraryName = "libcotengra";

libraryFunctions = LibraryFunctionLoad[
	$libraryName,
	"load_wstp_functions",
	LinkObject,
	LinkObject
][$libraryName];


GreedyPath[
	input : {{___String}...},
	output : {___String},
	sizeDict : KeyValuePattern[_String -> _Integer],
	costMod : _ ? NumericQ | None : None,
	temperature : _ ? NumericQ | None : None,
	simplify : True | False | None : None,
	useSSA : True | False | None : None
] := Block[{ds = Developer`DataStore, path},
	Enclose[
		path = List @@ List @@@ Confirm @ libraryFunctions["optimize_greedy"][
			ds @@ ds @@@ input,
			ds @@ output,
			ds @@ ds @@@ Normal[N /@ sizeDict],
			ds @ Replace[N[costMod], None -> Sequence[]],
			ds @ Replace[N[temperature], None -> Sequence[]],
			ds @ Replace[simplify, None -> Sequence[]],
			ds @ Replace[useSSA, None -> Sequence[]]
		];
		path + 1
	]
]

OptimalPath[
	input : {{___String}...},
	output : {___String},
	sizeDict : KeyValuePattern[_String -> _Integer],
	minimize : _String | None : None,
	costCap : _ ? NumericQ | None : None,
	searchOuter : True | False | None : None,
	simplify : True | False | None : None,
	useSSA : True | False | None : None
] := Block[{ds = Developer`DataStore, path},
	Enclose[
		path = List @@ List @@@ Confirm @ libraryFunctions["optimize_optimal"][
			ds @@ ds @@@ input,
			ds @@ output,
			ds @@ ds @@@ Normal[N /@ sizeDict],
			ds @ Replace[minimize, None -> Sequence[]],
			ds @ Replace[N[costCap], None -> Sequence[]],
			ds @ Replace[searchOuter, None -> Sequence[]],
			ds @ Replace[simplify, None -> Sequence[]],
			ds @ Replace[useSSA, None -> Sequence[]]
		];
		path + 1
	]
]


ContractIndices[i_, j_] := With[{c = Complement[Join[i, j], SymmetricDifference[i, j]]},
	c -> {DeleteElements[DeleteDuplicates[i], c], DeleteElements[DeleteDuplicates[j], c]}
]

TreePathToPath[treePath_List, indices : _List | Automatic : Automatic] := Block[{len, index, path = {}},
	index = Replace[indices, Automatic :> Sort[Cases[treePath, {x_} :> x, All]]];
	len = Length[index];
	index = AssociationThread[List /@ index, Range[len]];
	Scan[
		Block[{pos = Lookup[index, #], min, max, k},
			{min, max} = MinMax[pos];
			k = Length[pos];
			AppendTo[path, pos];
			index = Map[Which[# < min, #, # > max, # - k, True, # - 1] &, index];
			KeyDropFrom[index, #];
			AppendTo[index, # -> --len]
		] &,
		treePath,
		{0, -3}
	];
	path
]

PathToTreePath[path_List, indices : _List | Automatic : Automatic] :=
	First @ Fold[
		{idx, pos} |-> Append[
			Delete[idx, List /@ pos],
			If[Length[pos] == 1, idx[[pos[[1]]]], idx[[pos]]]
		],
		List /@ Replace[indices, Automatic :> Range[Count[path, {_, _}] + 1]],
		path
	]


CanonicalPath[path_List, indices : _List | Automatic : Automatic] :=
	TreePathToPath[PathToTreePath[path, indices], indices]

PathIndexContractions[path : {{_Integer, _Integer} ...}, indices : {__List}] :=
	DeleteCases[{}] @ FoldPairList[
		With[{c = ContractIndices @@ #1[[#2]]}, {c[[1]], Append[Delete[#1, List /@ #2], Catenate[c[[2]]]]}] &,
		indices,
		path
	]

PathIndexContractions[path_List, indices : {__List}, contractions : {__List}] :=
	With[{index = First /@ PositionIndex[Catenate[indices]]},
		Map[Lookup[index, #] &, PathIndexContractions[path, contractions], {3}]
	]

PathIndexContractions[path_List, KeyValuePattern[{"Indices" -> indices_, "Contractions" -> contractions_}]] :=
	Catenate @ PathIndexContractions[path, indices, contractions]


End[]
EndPackage[]