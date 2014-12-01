-module (shen_network).

%% API
-export ([start/3, train/2, test/2]).


%% ===================================================================
%% API
%% ===================================================================

start(NumAttrs, Classes, HiddenLayers) -> erlang:display(start_neurons), ok.

train(InputLayer, TrainSet) -> erlang:display(train), ok.

test(InputLayer, TestSet) -> erlang:display(test), ok.


%% ===================================================================
%% Internal Functions
%% ===================================================================



	% Network = [[spawn(neuron, start, [length(TrainSet)]) || X <- lists:seq(Dimension)] || Dimension <- NetworkArchitecture],

	% lists:foldl(fun(Layer, LayerAfter) -> lists:map(fun(Neuron) -> Neuron ! LayerAfter end), Layer end, [], Network),
	% lists:foldr(fun(Layer, LayerBefore) -> lists:map(fun(Neuron) -> Neuron ! LayerBefore end), Layer end, Layer, Network).

	% until convergence:

	% 	send the fist layer of neurons the first input




	% gradient descent loop, train network
			% get training data
			% test run and calculate accuracy


% Network = [[spawn(neuron, start, []) || X <- lists:seq(Dimension)] || Dimension <- NetworkArchitecture],

	% lists:foldl(fun(Layer, LayerAfter) -> lists:map(fun(Neuron) -> Neuron ! LayerAfter, Layer end, [], Network),
	% lists:foldr(fun(Layer, LayerBefore) -> lists:map(fun(Neuron) -> Neuron ! LayerBefore, Layer end, Layer, Network).

	% Make a first and last layer. 

	% First Layer
	% Find the dimensionality of the training data. 
	% Send appropriate Pids. 

	% Last layer
	% Find the dimenionality of the labels. 
	% Send the last hidden layer these Pids, and send these Pids the Layerbefore. 



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


