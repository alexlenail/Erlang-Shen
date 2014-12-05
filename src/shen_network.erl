-module (shen_network).
-export ([start/2]).



% NetworkArchitecture is a list of Hidden Layer Dimensions. 
start(DataInfo, NetworkArchitecture) -> 

	{TrainSet, TestSet} = shen_parse:get_data(DataInfo),

	Network = [[spawn(neuron, start, [length(TrainSet)]) || X <- lists:seq(Dimension)] || Dimension <- NetworkArchitecture],

	lists:foldr(fun(Layer, LayerBefore) -> lists:map(fun(Neuron) -> Neuron ! LayerBefore end), Layer end, self(), Network).
	lists:foldl(fun(Layer, LayerAfter) -> lists:map(fun(Neuron) -> Neuron ! LayerAfter end), Layer end, self(), Network),

	until convergence:

		send the fist layer of neurons the first input

		once they receive it, they will forward propagate it until the end of the chain. 

		the last neuron will send its activation to the Network
		the Network compares the true value agains the Networks output. 
		sends delta back to the last layer with theta value of 1. 

		they backpropagate the value to the first layer. 
		once the values are at the before last layer, 




	receive
		{LastLayer, Activation} -> 
			LastLayer ! CorrectAnswer,

	end.

% bias units?


