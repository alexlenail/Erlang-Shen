-module(shen).

-behaviour(application).

%% Application callbacks
-export([start/2, stop/1]).


%% ===================================================================
%% Application callbacks
%% ===================================================================

start(_StartType, [TrainSet, TestSet, HiddenLayers]) ->
	shen_sup:start_link(),
	case shen_parse:arff(TrainSet) of
		{error, _Reason} -> {error, "Invalid training data file"};
		{ok, NumAttrs, Classes, Instances} ->
			InputLayer = spawn_neurons(NumAttrs, Classes)
			% gradient descent loop, train network
			% get training data
			% test run and calculate accuracy
	end.

stop(_State) ->
    ok.
    % clean up whatever needs to be cleaned


%% ===================================================================
%% Internal Logic
%% ===================================================================

% spawn neurons and return pid list of input layer
% maybe be need number of training instances?
spawn_neurons(NumAttrs, Classes) -> ok.
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


% bias units?
