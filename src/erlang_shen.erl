-module(erlang_shen).

-behaviour(application).

%% Application callbacks
-export([start/2, stop/1]).

%% ===================================================================
%% Application callbacks
%% ===================================================================

start(_StartType, _StartArgs) ->
    erlang_shen_sup:start_link().
    % get the training data specified by user to train our network
    % start all the neurons
        % maybe get network architecture from user input or generate ourselves based on training data type
        % get back input layer pids as list
    % send training data sequentially to input layer and train network
        % perform backprop and gradient checking etc
    % get test data and send through network


stop(_State) ->
    ok.
    % clean up whatever needs to be cleaned
