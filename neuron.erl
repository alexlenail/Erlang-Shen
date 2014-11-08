-module (neuron).
% -import (math, [exp/1]).
-export ([init/2]).

% LayerBefore and LayerAfter are PID lists. 
init(LayerBefore, LayerAfter) -> 
	random initialization of Theta
	loop().

loop(LayerBefore, LayerAfter, Theta) -> 

	receive
		{Pid from LayerBefore, TrainingExampleID, Value} -> forward(Value);
		{Pid from LayerAfter}, TrainingExampleID, Value} -> backprop(Value);
	end,
	loop().

% Theta a list of [weights] k dimensional where k is the dimension of the layer before
forward(Value, Theta) ->

	LayerAfter ! {self(), TrainingExampleID, g(linear combination of Theta and Values)



g(Z) -> 1/(1+math:exp(-Z))


=== 
backprop() -> 

compute J(theta)
send that to all nodes in the past. 


=== 
% gradient checking?


