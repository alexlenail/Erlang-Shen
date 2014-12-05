-module(shen).
-behaviour(application).

%% API
-export([run/3]).

%% Application Callbacks
-export([start/2, stop/1]).


%% ===================================================================
%% API Functions
%% ===================================================================

run(TrainSet, TestSet, HiddenLayerDims) ->
	case shen_parse:arff(TrainSet) of
		{error, _Reason} ->
			{error, invalid_training_data};
		{ok, {NumAttrs, Classes, TrainInstances}} ->
			case shen_parse:arff(TestSet) of
				{ok, {NumAttrs, Classes, TestInstances}} ->
					Layers = shen_network:build(NumAttrs, Classes, HiddenLayerDims),
					shen_network:train(Layers, TrainInstances),
					shen_network:test(Layers, TestInstances);
				_TestDataError ->
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
