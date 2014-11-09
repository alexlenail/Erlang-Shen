-module (neuron).
% -import (math, [exp/1]).
-import (lists, [member/2]).
-export ([init/2]).

% LayerBefore and LayerAfter are PID lists. 
init(LayerBefore, LayerAfter) -> 
	random initialization of Theta

	linear combination = fun(Activations) -> linear comb(Activations, Theta)

buildCurry(Len, Sofar) when Len =:= 0 -> Sofar;
buildCurry(Len, Sofar) when Len > 0 -> fun(X) -> Sofar([X|]) end;

	buildCurry(length(LayerBefore), linear combination. )

	fun(A) -> fun(B) -> fun(C) -> fun(D) -> fun([Activations]) -> linear comb(Activations, Theta)


	loop(LayerBefore, LayerAfter, Theta).


loop(LayerBefore, LayerAfter, Theta) -> 

	receive
		{Pid, Activation} when member(Pid, LayerBefore) -> forward(Activation, Theta);
		{Pid, Error} when member(Pid, LayerAfter) -> backprop(Error);
	end,

	loop().

% Theta a list of [weights] k dimensional where k is the dimension of the layer before
forward(Activation, Theta) when is last activation ->

	buildCurry(Activation)


	LayerAfter ! {self(), TrainingExampleID, g(linear combination of Theta and Values)}



g(Z) -> 1/(1+math:exp(-Z))


fun(A) -> fun(B) -> fun(C) -> fun(D) -> LayerAfter ! {self(), TrainingExampleID, g(linear combination of Theta and [A, B, C, D])}



=== 
backprop() -> 

compute J(theta)
send that to all nodes in the past. 


=== 
% gradient checking?


