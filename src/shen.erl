%% ===================================================================
%% shen.erl
%%
%% This is the main module that implements an Erlang OTP application
%% behaviour. It implements an API that users can interact with to
%% provide parameters and specify data to run the algorithm on.
%%
%% ===================================================================

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
	shen_print:title("Parsing ~s~n", [TrainSet]),
	case shen_parse:arff(TrainSet) of
		{error, _Reason1} ->
			shen_print:event("Error: Invalid training data file ~s~n", [TrainSet]);
		{ok, {NumAttrs, Classes, TrainInstances}} ->
			shen_print:event("Done~n", []),
			shen_print:title("Parsing ~s~n", [TestSet]),
			case shen_parse:arff(TestSet) of
				{ok, {NumAttrs, Classes, TestInstances}} ->
					shen_print:event("Done~n", []),
					Layers = shen_network:build(NumAttrs, Classes, HiddenLayerDims),
					shen_print:title("Training Network~n", []),
					shen_network:train(NumIterations, Layers, TrainInstances, TestInstances),
					{InputLayer, _, _} = Layers,
					shen_network:test(NumIterations, HiddenLayerDims, InputLayer, TrainSet, TestSet, TestInstances),
					shen_network:finish(Layers);
				{error, _Reason2} ->
					shen_print:event("Error: Invalid test data file ~s~n", [TestSet])
			end
	end.


%% ===================================================================
%% Application Callbacks
%% ===================================================================

start(_StartType, _Args) ->
	shen_print:title("Shen application starting~n", []),
	shen_sup:start_link().

stop(_State) ->
    ok.
