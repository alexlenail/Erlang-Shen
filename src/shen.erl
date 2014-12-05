-module(shen).
-behaviour(application).

%% API
-export([run/3]).

%% Application Callbacks
-export([start/2, stop/1]).


%% ===================================================================
%% API Functions
%% ===================================================================

run(TrainSet, TestSet, HiddenLayers) ->
	case shen_parse:arff(TrainSet) of
		{error, _Reason} ->
			{error, invalid_training_data};
		{ok, {NumAttrs, Classes, TrainInstances}} ->
			case shen_parse:arff(TestSet) of
				{ok, {NumAttrs, Classes, TestInstances}} ->
					InputLayer = shen_network:build(NumAttrs, Classes, HiddenLayers),
					shen_network:train(InputLayer, TrainInstances),
					shen_network:test(InputLayer, TestInstances);
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
