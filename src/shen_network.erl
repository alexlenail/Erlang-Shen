-module (shen_network).

%% API
-export ([build/3, train/4, test/2, finish/1]).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
build(NumAttrs, _Classes, HiddenLayerDims) ->
	case whereis(?MODULE) of
		undefined -> register(?MODULE, self());
		_Pid -> ok
	end,
	% start input layer
	{ok, InputLayer} = start_layer(NumAttrs, input),
	% start hidden layers
	HiddenLayers = lists:map(fun(LayerSize) ->
									erlang:display(start_layer),
									{ok, LayerPids} = start_layer(LayerSize, hidden),
									erlang:display(LayerPids),
									LayerPids
								end,
								HiddenLayerDims),
	% start output layer
	{ok, OutputLayer} = start_layer(1, output),
	% start collector process for hidden bias units and gradient descent
	% start_collector(length(InputLayer ++ lists:flatten(HiddenLayers))),
	connect_layers(InputLayer, HiddenLayers, OutputLayer),
	{InputLayer, HiddenLayers, OutputLayer}.

train(0, _Layers, _TrainSet, _TestSet) -> erlang:display(done_training), ok;
train(NumGradientSteps, {InputLayer, HiddenLayers, OutputLayer}, TrainSet, TestSet) ->
	% generate truly random numbers
    <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
    random:seed(A, B, C),
	Shuffled = shuffle_instances(TrainSet),
	% train on each instance in random order
	lists:map(fun(Inst) -> train_instance(InputLayer, Inst) end, Shuffled),
	% once we have finished training on the whole set, tell each neuron
	% in input and hidden layers to use gradient descent and compute new thetas
	lists:map(fun(Pid) ->
				gen_server:cast(Pid, {descend_gradient, length(TrainSet)})
			  end,
			  InputLayer ++ lists:flatten(HiddenLayers)),

	train_iteration_end(length(InputLayer ++ lists:flatten(HiddenLayers))),
	
	test(InputLayer, TestSet), % remove this when done

	train(NumGradientSteps-1, {InputLayer, HiddenLayers, OutputLayer}, TrainSet, TestSet).

test(InputLayer, TestSet) ->
	% Error = lists:sum(lists:map(fun(Inst) -> test_instance(InputLayer, Inst) end, TestSet))/length(TestSet) * 100,
	% io:format("~i%~n", [Error]),
	E = lists:map(fun(Inst) -> test_instance(InputLayer, Inst) end, TestSet),
	erlang:display(E),
	ok.

finish({InputLayer, HiddenLayers, OutputLayer}) ->
	% end all neuron processes
	lists:map(fun(Pid) ->
				shen_sup:end_child(Pid)
			  end,
			  InputLayer ++ lists:flatten(HiddenLayers) ++ OutputLayer),
	unregister(?MODULE),
	ok.


%% ===================================================================
%% Internal Functions
%% ===================================================================

% starts LayerSize neurons and returns a list of ther Pids
start_layer(LayerSize, Type) -> start_layer(LayerSize, Type, []).
start_layer(0, _Type, Pids) -> {ok, Pids};
start_layer(LayerSize, Type, Pids) ->
	% start new neuron with type
	case shen_sup:start_child({neuron_type, Type}) of
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
				[?MODULE], % connect network to the input layer
				[InputLayer] ++ HiddenLayers ++ [OutputLayer]),
	% connect layers backwards
	lists:foldr(fun(Layer, LayerAfter) ->
					lists:map(fun(NeuronPid) ->
								gen_server:cast(NeuronPid, {layer_after, LayerAfter})
							  end,
							  Layer),
					Layer
				end,
				[?MODULE], % connect network to the output layer
				[InputLayer] ++ HiddenLayers ++ [OutputLayer]),
	ok.

train_instance(InputLayer, Inst) ->
	Label = lists:last(Inst),
	% map across inputlayer, send 1 feature to a node
	erlang:display({start_forwardprop, Inst}),
	lists:map(fun({Pid, Attr}) ->
				gen_server:cast(Pid, {forwardprop, network, Attr})
			  end,
			  lists:zip(InputLayer, lists:droplast(Inst))),
	% receive message from output layer that we are done with forward
	receive
		{forwardprop, OutputPid, _Prediction} -> % send output layer actual class
			erlang:display({network_pred, _Prediction}),
			erlang:display(start_backprop),
			gen_server:cast(OutputPid, {backprop, network, Label})
	end,
	train_instance_end(length(InputLayer)).

% receive messages from input layer saying network is done with this instance
train_instance_end(0) -> ok;
train_instance_end(N) ->
	receive
		finished_instance ->
			train_instance_end(N-1)
	end.

train_iteration_end(0) -> ok;
train_iteration_end(NumNeurons) ->
	receive
		finished_descent -> train_iteration_end(NumNeurons - 1)
	end.

test_instance(InputLayer, Inst) ->
	Label = lists:last(Inst),
	% map across inputlayer, send 1 feature to a node
	lists:map(fun({Pid, Attr}) ->
				gen_server:cast(Pid, {forwardprop, network, Attr})
			  end,
			  lists:zip(InputLayer, lists:droplast(Inst))),
	% receive message from output layer that we are done with forward
	receive
		{forwardprop, _OutputPid, Prediction} -> Label - Prediction
	end.

% shuffle list of data instances in random order
shuffle_instances([]) -> [];
shuffle_instances([X]) -> [X];
shuffle_instances(Xs) -> shuffle_instances(Xs, length(Xs), []).
shuffle_instances([], 0, Shuffled) -> Shuffled;
shuffle_instances(Xs, Len, Shuffled) ->
    {X, Rest} = nth_rest(random:uniform(Len), Xs),
    shuffle_instances(Rest, Len - 1, [X | Shuffled]).

% pick a random element from list and return it and rest
nth_rest(N, List) -> nth_rest(N, List, []).
nth_rest(1, [E|List], Prefix) -> {E, Prefix ++ List};
nth_rest(N, [E|List], Prefix) -> nth_rest(N - 1, List, [E|Prefix]).
