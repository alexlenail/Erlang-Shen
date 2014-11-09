-module (neuron).
% -import (math, [exp/1]).
-import (lists, [member/2]).
-export ([init/2]).

% LayerBefore and LayerAfter are PID lists. 
init(LayerBefore, LayerAfter) -> 
	random initialization of Theta
	loop(LayerBefore, LayerAfter, Theta).

loop(LayerBefore, LayerAfter, Theta) -> 

	receive
		{Pid, Activation} when member(Pid, LayerBefore) -> forward(Activation, Theta);
		{Pid, Error} when member(Pid, LayerAfter) -> backprop(Error);
	end,

	loop().

% Theta a list of [weights] k dimensional where k is the dimension of the layer before
forward(Activation, Theta) when is last activation ->

	LayerAfter ! {self(), TrainingExampleID, g(linear combination of Theta and Values)}



g(Z) -> 1/(1+math:exp(-Z))


=== 
backprop() -> 

compute J(theta)
send that to all nodes in the past. 


=== 
% gradient checking?


