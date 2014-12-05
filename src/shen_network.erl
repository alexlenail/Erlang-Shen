-module (shen_network).

%% API
-export ([build/3, train/2, test/2]).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
build(NumAttrs, Classes, HiddenLayerDims) ->
	% start input layer
	{ok, InputLayer} = start_layer(NumAttrs, input),
	% start hidden layers
	HiddenLayers = lists:map(fun(LayerSize) ->
									{ok, LayerPids} = start_layer(LayerSize, hidden),
									LayerPids
								end,
								HiddenLayerDims),
	% start output layer
	{ok, OutputLayer} = start_layer(1, output),
	connect_layers(InputLayer, HiddenLayers, OutputLayer),
	% return input layer so we can send it instances
	InputLayer.

train(InputLayer, TrainSet) ->
	erlang:display(train),

	% loop until convergence
		% shuffle the training examples
		% for each example
			% map across inputlayer, send 1 feature to a node
			% receive message from output layer that we are done with forward
			% send output layer actual class
			% receive messages from input layer saying they are done
			

	ok.

test(InputLayer, TestSet) ->
	erlang:display(test),
	ok.


%% ===================================================================
%% Internal Functions
%% ===================================================================

% starts LayerSize neurons and returns a list of ther Pids
start_layer(LayerSize, Type) -> start_layer(LayerSize, Type, []).
start_layer(0, Type, Pids) -> {ok, Pids};
start_layer(LayerSize, Type, Pids) ->
	% start new neuron with type and access to parent and link to supervisor
	Args = {{network_pid, self()}, {neuron_type, Type}},
	case shen_sup:start_child(Args) of
		{error, Reason} -> {error, Reason};
		{ok, ChildPid} ->
			start_layer(LayerSize-1, Type, [ChildPid | Pids])
	end.

connect_layers(InputLayer, HiddenLayers, OutputLayer) ->
	% connect layers forward
	lists:foldl(fun(Layer, LayerBefore) ->
					lists:map(fun(NeuronPid) ->
								gen_server:cast(NeuronPid, {layer_before, LayerBefore})
							  end,
							  Layer),
					Layer
				end,
				[self()], % connect network to the input layer
				lists:append([[InputLayer], HiddenLayers, [OutputLayer]])),
	% connect layers backwards
	lists:foldr(fun(Layer, LayerAfter) ->
					lists:map(fun(NeuronPid) ->
								gen_server:cast(NeuronPid, {layer_after, LayerAfter})
							  end,
							  Layer),
					Layer
				end,
				[self()],
				lists:append([[InputLayer], HiddenLayers, [OutputLayer]])),
	ok.

% forward_back_once([], []) -> 



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
