-module (shen_network).

%% API
-export ([build/3, train/2, test/2]).

-define(LAMBDA, .0005).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
build(NumAttrs, Classes, HiddenLayerDims) ->
	register(self(), network),
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
	% start collector process for hidden bias units and gradient descent
	start_collector(),
	connect_layers(InputLayer, HiddenLayers, OutputLayer),
	{InputLayer, HiddenLayers, OutputLayer}.

train({InputLayer, HiddenLayers, OutputLayer} TrainSet) ->
	% generate truly random numbers
    <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
    random:seed(A, B, C),

	Shuffled = shuffle_instances(TrainSet),
	lists:map(fun train_instance/1, Shuffled),

	collector ! {getAccumulatedError, length(TrainSet)},
	receive
		{accumulatedBiasErrorList, BiasList} -> ok
	end,
	% getting thetas and deltas from each neuron
	lists:flatmap(fun(Pid) ->
					gen_server:cast(Pid, {descend_gradient, length(TrainSet), Bias})
				  end,
				  [InputLayer] ++ HiddenLayers ++ [OutputLayer]),



			% all the deltas from the net

		% compute partial derivatives
		% perform update
		% send every neuron updated Thetas
		% loop on train. 

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

% =====================================================

start_collector() ->
	CollectorPid = spawn(fun() -> collector(maps:new(), maps:new(), maps:new()) end),
	register(collector, CollectorPid).

collector(BiasMap) ->
	receive
		{bias, Pid, X} ->
			case find(Pid, BiasMap) of
				{ok, V} -> bias_accum(maps:put(Pid, V+X, BiasMap), Dij);
				error -> bias_accum(maps:put(Pid, X, BiasMap), Dij)
			end;
		{getAccumulatedError, M} -> 
			network ! {accumulatedBiasErrorList, lists:map(fun({Pid, Bias}) -> {Pid, Bias/M} end, maps:to_list(BiasMap))}, 
			bias_accum(maps:new());
		stop -> ok
	end.


% =====================================================

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
				[InputLayer] ++ HiddenLayers ++ [OutputLayer]),
	% connect layers backwards
	lists:foldr(fun(Layer, LayerAfter) ->
					lists:map(fun(NeuronPid) ->
								gen_server:cast(NeuronPid, {layer_after, LayerAfter})
							  end,
							  Layer),
					Layer
				end,
				[self()],
				[InputLayer] ++ HiddenLayers ++ [OutputLayer]),
	ok.

train_instance(InputLayer, Inst) ->
	Label = lists:last(Inst),
	% map across inputlayer, send 1 feature to a node
	lists:map(fun({Pid, Attr}) ->
				gen_server:cast(Pid, {forwardprop, network, Attr})
			  end,
			  lists:zip(InputLayer, lists:droplast(Inst)),
	% receive message from output layer that we are done with forward
	receive
		{forwardprop, OutputPid, Prediction} -> % send output layer actual class
			OutputPid ! {backprop, network, Label}
	end,
	% receive messages from input layer saying netowrk is done with this instance
	train_instance_receive_end(length(InputLayer)).

train_instance_receive_end(0) -> ok.
train_instance_receive_end(N) ->
	receive
		{finished, NewDeltas} -> erlang:display(NewDeltas), train_instance_receive_end(N-1)
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
