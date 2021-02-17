%% ===================================================================
%% shen_network.erl
%%
%% Builds and maintains the relationships between layers. The API
%% also implements train, which runs gradient descent the number
%% of times specified by the user on the training data; test, which
%% runs the test data on the network and outputs results; and finish
%% which ends all child neurons and shuts down the network.
%%
%% ===================================================================

-module (shen_network).

%% API
-export ([build/3, train/4, test/6, finish/1, test_instance/3]).


%% ===================================================================
%% API Functions
%% ===================================================================

% starts network specified by NumAttrs and HiddenLayers parameters
build(NumAttrs, _Classes, HiddenLayerDims) ->
	shen_print:title("Building Network~n", []),
	case whereis(?MODULE) of
		undefined -> register(?MODULE, self());
		_Pid -> ok
	end,
	% start input layer
	shen_print:event("Starting input layer~n", []),
	{ok, InputLayer} = start_layer(NumAttrs, input),
	% start hidden layers
	HiddenLayers = lists:map(fun(LayerSize) ->
									shen_print:event("Starting hidden layer of size ~w~n", [LayerSize]),
									{ok, LayerPids} = start_layer(LayerSize, hidden),
									LayerPids
								end,
								HiddenLayerDims),
	% start output layer
	shen_print:event("Starting output layer~n", []),
	{ok, OutputLayer} = start_layer(1, output),
	connect_layers(InputLayer, HiddenLayers, OutputLayer),
	shen_print:event("Done~n", []),
	{InputLayer, HiddenLayers, OutputLayer}.

train(0, _Layers, _TrainSet, _TestSet) ->
	shen_print:event("Done~n", []);
train(NumGradientSteps, {InputLayer, HiddenLayers, OutputLayer}, TrainSet, TestSet) ->
	% generate truly random numbers
	Rand = crypto:strong_rand_bytes(12),
	<<A:32, B:32, C:32>> = Rand,
    rand:seed(exsplus, {A, B, C}),
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
	shen_print:event("Gradient descent step complete, ~w to go~n", [NumGradientSteps-1]),
	train(NumGradientSteps-1, {InputLayer, HiddenLayers, OutputLayer}, TrainSet, TestSet).

test(NumGradientSteps, HiddenLayerDims, InputLayer, TrainFileName, TestFileName, TestSet) ->
	shen_print:title("Running test data~n", []),
	OutFileName = gen_outfile_name(TestFileName),
	{ok, OutFile} = file:open(OutFileName, [write]),
	file:write(OutFile, io_lib:fwrite("Training Data: ~s~n",[TrainFileName])),
	file:write(OutFile, io_lib:fwrite("Test Data: ~s~n",[TestFileName])),
	file:write(OutFile, io_lib:fwrite("Hidden Layer Architecture: ~w~n",[HiddenLayerDims])),
	file:write(OutFile, io_lib:fwrite("Gradient Descent Steps: ~w~n",[NumGradientSteps])),
	file:write(OutFile, io_lib:fwrite("~nInstances and Predictions~n",[])),
	TestResults = lists:map(fun(Inst) -> test_instance(InputLayer, Inst, OutFile) end, TestSet),
	file:close(OutFile),
	NumCorrect = lists:foldl(fun({L, P}, A) ->
								case L =:= P of
									true -> A + 1;
									false -> A
								end
							end,
							0,
							TestResults),
	shen_print:event("Results output to ~s~n", [OutFileName]),
	shen_print:event("Accuracy: ~w%~n", [100 * (NumCorrect / length(TestSet))]).

finish({InputLayer, HiddenLayers, OutputLayer}) ->
	shen_print:title("Shutting down network~n", []),
	shen_print:event("Ending neuron processes~n", []),
	% end all neuron processes
	lists:map(fun(Pid) ->
				shen_sup:end_child(Pid)
			  end,
			  InputLayer ++ lists:flatten(HiddenLayers) ++ OutputLayer),
	unregister(?MODULE),
	shen_print:event("Done~n", []).


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

% sends neurons in each layer the Pids of the layer before and layer after
% input layer gets network as its layer before, output layer gets network as its layer after
connect_layers(InputLayer, HiddenLayers, OutputLayer) ->
	shen_print:event("Connecting layers~n", []),
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
	lists:map(fun({Pid, Attr}) ->
				gen_server:cast(Pid, {forwardprop, network, Attr})
			  end,
			  lists:zip(InputLayer, lists:droplast(Inst))),
	% receive message from output layer that we are done with forward
	receive
		{forwardprop, OutputPid, _Prediction} -> % send output layer actual class
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

test_instance(InputLayer, Inst, OutFile) ->
	Label = lists:last(Inst),
	Attributes = lists:droplast(Inst),
	% map across inputlayer, send 1 feature to a node
	lists:map(fun({Pid, Attr}) ->
				gen_server:cast(Pid, {forwardprop, network, Attr})
			  end,
			  lists:zip(InputLayer, Attributes)),
	% receive message from output layer that we are done with forward
	receive
		{forwardprop, _OutputPid, Activation} ->
			% take raw prediction and map to 0 or 1
			case Activation >= 0.55 of
				true -> Prediction = 1;
				false -> Prediction = 0
			end,
			StringifiedAttrs = lists:map(fun(Attr) ->
											case is_float(Attr) of
												true -> float_to_list(Attr, [{decimals, 10}, compact]);
												false -> integer_to_list(Attr)
											end
										end,
										Attributes),
			OutLine = string:join(StringifiedAttrs ++ [integer_to_list(Label)] ++ [integer_to_list(Prediction)], ","),
			% output instance+prediction to outfile
			case OutFile =:= [] of
				true -> ignore;
				false -> file:write(OutFile, io_lib:fwrite("~s~n",[OutLine]))
			end,
			{Label, Prediction}
	end.

gen_outfile_name(TestFileName) ->
	% get timestamp of now
	{{Year, Month, Day}, {Hour, Minute, Second}} = erlang:localtime(),
	TimeStampString = integer_to_list(Year) ++ "-" ++ integer_to_list(Month) ++ "-" ++
					  integer_to_list(Day) ++ "-" ++ integer_to_list(Hour) ++ ":" ++
					  integer_to_list(Minute) ++ ":" ++ integer_to_list(Second),
	% strip directories before file name and .arff
	StrippedTestFileName = re:replace(lists:last(string:tokens(TestFileName, "/")), ".arff", "", [global, {return, list}]),
	"results/shen_results_" ++ StrippedTestFileName ++ "_" ++ TimeStampString ++ ".txt".

% shuffle list of data instances in random order
shuffle_instances([]) -> [];
shuffle_instances([X]) -> [X];
shuffle_instances(Xs) -> shuffle_instances(Xs, length(Xs), []).
shuffle_instances([], 0, Shuffled) -> Shuffled;
shuffle_instances(Xs, Len, Shuffled) ->
    {X, Rest} = nth_rest(rand:uniform(Len), Xs),
    shuffle_instances(Rest, Len - 1, [X | Shuffled]).

% pick a random element from list and return it and rest
nth_rest(N, List) -> nth_rest(N, List, []).
nth_rest(1, [E|List], Prefix) -> {E, Prefix ++ List};
nth_rest(N, [E|List], Prefix) -> nth_rest(N - 1, List, [E|Prefix]).
