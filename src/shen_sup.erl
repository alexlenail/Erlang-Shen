-module(shen_sup).
-behaviour(supervisor).

%% API
-export([start_link/0]).

%% Supervisor Callbacks
-export([init/1]).


-define(SERVER, ?MODULE).

%% Helper macro for declaring children of supervisor
-define(CHILD(I, Type), {I, {I, start_link, []}, permanent, 5000, Type, [I]}).


%% ===================================================================
%% API Functions
%% ===================================================================

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).


%% ===================================================================
%% Supervisor Callbacks
%% ===================================================================

init(_Args) ->
    %%%%%%%%%%%%%%%%%%%%%%%% NEED TO IRON THIS OUT, JUST A PLACEHOLDER %%%%%%%%%%%%%%%%%%%%%%
    {ok, {{one_for_one, 5, 10}, []}}.
    % RestartStrategy = one_for_all,
    % MaxRestarts = 1000,
    % MaxSecondsBetweenRestarts = 3600,

    % SupFlags = {RestartStrategy, MaxRestarts, MaxSecondsBetweenRestarts},

    % Restart = permanent,
    % Shutdown = 2000,
    % Type = worker,

    % AChild = {'AName', {'AModule', start_link, []},
    %           Restart, Shutdown, Type, ['AModule']},

    % {ok, {SupFlags, [AChild]}}.
