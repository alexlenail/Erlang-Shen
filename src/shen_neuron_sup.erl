-module(shen_neuron_sup).

-behaviour(supervisor).

%% API
-export([start_link/0]).

%% Supervisor callbacks
-export([init/1]).

%% Helper macro for declaring children of supervisor
% -define(CHILD(I, Type), {I, {I, start_link, []}, permanent, 5000, Type, [I]}).


%% ===================================================================
%% API functions
%% ===================================================================

start_link() -> ok.
    % supervisor:start_link({local, ?MODULE}, ?MODULE, []).
    % take params of # neurons and layers and start_link each one


%% ===================================================================
%% Supervisor callbacks
%% ===================================================================

init([]) ->
    {ok, {{one_for_one, 5, 10}, []}}.
    % probably change to one_for_all so we can kill program if any neuron dies