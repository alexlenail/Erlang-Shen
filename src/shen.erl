-module(shen).
-behaviour(application).

%% API
-export([run/4]).

%% Application Callbacks
-export([start/2, stop/1]).


%% ===================================================================
%% API Functions
%% ===================================================================

run(TrainSet, TestSet, HiddenLayerDims, NumIterations) ->
	case shen_parse:arff(TrainSet) of
		{error, _Reason1} ->
			{error, invalid_training_data};
		{ok, {NumAttrs, Classes, TrainInstances}} ->
			case shen_parse:arff(TestSet) of
				{ok, {NumAttrs, Classes, TestInstances}} ->
					Layers = shen_network:build(NumAttrs, Classes, HiddenLayerDims),
					shen_network:train(NumIterations, Layers, TrainInstances, TestInstances),
					shen_network:test(Layers, TestInstances);
				{error, _Reason2} ->
					{error, invalid_test_data}
			end
	end.


%% ===================================================================
%% Application Callbacks
%% ===================================================================

start(_StartType, _Args) ->
	shen_sup:start_link().

stop(_State) ->
    ok. % clean up whatever needs to be cleaned
