-module(shen_sup).
-behaviour(supervisor).

%% API
-export([start_link/0, start_child/1]).

%% Supervisor Callbacks
-export([init/1]).

-define(SERVER, ?MODULE).

%% ===================================================================
%% API Functions
%% ===================================================================

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

start_child(Args) ->
    supervisor:start_child(?MODULE, [Args]).


%% ===================================================================
%% Supervisor Callbacks
%% ===================================================================

init(_Args) ->
    %%%%%%% REFINE THESE PARAMETERS %%%%%%%
    RestartStrategy = simple_one_for_one,
    MaxRestarts = 1000,
    MaxSecondsBetweenRestarts = 3600,
    SupFlags = {RestartStrategy, MaxRestarts, MaxSecondsBetweenRestarts},
    Restart = permanent,
    Shutdown = brutal_kill, % 2000,
    Type = worker,
    AChild = {shen_neuron, {shen_neuron, start_link, []},
               Restart, Shutdown, Type, [shen_neuron]},
    {ok, {SupFlags, [AChild]}}.
