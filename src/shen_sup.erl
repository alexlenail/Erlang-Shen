%% ===================================================================
%% shen_sup.erl
%%
%% Serves as the supervisor for all of the spawned child processes.
%% Implements an API that allows the application process to link the
%% supervisor and register it, as well as safely start and manage the
%% restart strategies for neurons.
%%
%% ===================================================================

-module(shen_sup).
-behaviour(supervisor).

%% API
-export([start_link/0, start_child/1, end_child/1]).

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

end_child(Pid) ->
    supervisor:terminate_child(?MODULE, Pid).


%% ===================================================================
%% Supervisor Callbacks
%% ===================================================================

init(_Args) ->
    RestartStrategy = simple_one_for_one,
    MaxRestarts = 1,
    MaxSecondsBetweenRestarts = 5,
    SupFlags = {RestartStrategy, MaxRestarts, MaxSecondsBetweenRestarts},
    Restart = permanent,
    Shutdown = brutal_kill,
    Type = worker,
    AChild = {shen_neuron, {shen_neuron, start_link, []},
               Restart, Shutdown, Type, [shen_neuron]},
    {ok, {SupFlags, [AChild]}}.
