-module (shen_network).

%% API
-export ([build/3, train/2, test/2]).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
build(NumAttrs, Classes, HiddenLayerDims) ->
	% this solved a bunch of issues. 
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
	% start bias accumulator process
	start_bias(),
	connect_layers(InputLayer, HiddenLayers, OutputLayer),
	% return input layer so we can send it instances
	InputLayer.

train(InputLayer, TrainSet) ->
	% generate truly random numbers
    <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
    random:seed(A, B, C),

	% loop until convergence
		Shuffled = shuffle_instances(TrainSet),
		lists:map(fun train_instance/1, Shuffled),
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

start_bias() ->
	BiasPid = spawn(fun() -> bias_accum(maps:new()) end),
	register(bias, BiasPid).

bias_accum(BiasMap) ->
	receive
		{update, Pid, X} ->
			case find(Pid, BiasMap) of
				{ok, V} -> bias_accum(maps:put(Pid, V+X, BiasMap));
				error -> bias_accum(maps:put(Pid, X, BiasMap))
			end;
		{getAccumulatedError, Pid} -> Pid ! BiasMap, bias_accum(maps:new());
		stop -> ok
	end.

% we have bias, Now we need to figure out how to integrate it into computations

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

% map across inputlayer, send 1 feature to a node
% receive message from output layer that we are done with forward
% send output layer actual class
% receive messages from input layer saying they are done
train_instance(InputLayer, Inst) ->
	Label = stringToInt(lists:last(Inst)),
	lists:map(fun(Pid, Feature) -> gen_server:cast(Pid, {forwardprop, Feature}) end, lists:zip(InputLayer, lists:droplast(Inst))
	receive
		{forwardprop, PrevPid, Prediction} -> PrevPid !  {backprop, Prediction - Label}



	% erlang:display(Inst).

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
