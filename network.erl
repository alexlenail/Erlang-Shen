-module (network).
-export ([function/arity]).



% NetworkArchitecture is a list of Hidden Layer Dimensions. 
init(Dataset, NetworkArchitecture) -> 

	Network = [[spawn(neuron, start, []) || X <- lists:seq(Dimension)] || Dimension <- NetworkArchitecture]

	lists:foldl(fun(Layer, LayerAfter) -> lists:map(fun(Neuron) -> Neuron ! LayerAfter, Layer), Layer. end, Network)
	lists:foldr(fun(Layer, LayerBefore) -> lists:map(fun(Neuron) -> Neuron ! LayerBefore, Layer), Layer. end, Network)

	% Make a first and last layer. 

	% First Layer
	% Find the dimensionality of the training data. 
	% Send appropriate Pids. 

	% Last layer
	% Find the dimenionality of the labels. 
	% Send the last hidden layer these Pids, and send these Pids the Layerbefore. 



% learn
    % send data to input layer


% bias units?


