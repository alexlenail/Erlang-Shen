-module (shen_network).

%% API
-export ([start/3, train/2, test/2]).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
start(NumAttrs, Classes, HiddenLayers) ->
	erlang:display(start_neurons),
	% start input layer
	{ok, InputLayer} = start_layer(NumAttrs),
	erlang:display(InputLayer),
	% start hidden layers
	HiddenLayerPids = lists:map(fun(LayerSize) ->
									{ok, LayerPids} = start_layer(LayerSize),
									LayerPids
								end,
								HiddenLayers),
	erlang:display(HiddenLayerPids),
	% start output layer
	{ok, OutputLayer} = start_layer(1),
	% connect layers forwards

	% connect layers backwards

	% return input layer so we can send it instances
	InputLayer.

train(InputLayer, TrainSet) ->
	erlang:display(train),
	ok.

test(InputLayer, TestSet) ->
	erlang:display(test),
	ok.


%% ===================================================================
%% Internal Functions
%% ===================================================================

% starts LayerSize neurons and returns a list of ther Pids
start_layer(LayerSize) -> start_layer(LayerSize, []).
start_layer(0, Pids) -> {ok, Pids};
start_layer(LayerSize, Pids) ->
	% start new neuron and link to supervisor
	case shen_sup:start_child() of
		{error, Reason} -> {error, Reason};
		{ok, ChildPid} ->
			% gen_server:cast(ChildPid, Type). ??????
			start_layer(LayerSize-1, [ChildPid | Pids])
	end.





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


