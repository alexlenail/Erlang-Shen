-module(shen_neuron).
-behaviour(gen_server).

%% API
-export([start_link/1]).

%% Gen Server Callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-define(INIT_EPSILON, 0.0001).


%% ===================================================================
%% API Functions
%% ===================================================================

start_link(Args) ->
    gen_server:start_link(?MODULE, Args, []).


%% ===================================================================
%% Gen Server Callbacks
%% ===================================================================

-record(neuron, {network_pid, type, layer_before, layer_after,
				 inputs = [], activation, thetas, bias_theta, delta = 0, deltas}).

init({{network_pid, NetworkPid}, {neuron_type, Type}}) ->
	io:format("init neuron (~w)~n", [self()]),
	InitState = #neuron{network_pid = NetworkPid, type = Type, deltas = maps:new()},
	erlang:display(InitState),
    {ok, InitState}.

% asynchronous messages
handle_cast(Msg, State) ->
	NewState = update(Msg, State),
    erlang:display(NewState),
    {noreply, NewState}.

handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.


%% ===================================================================
%% Internal Functions
%% ===================================================================

% this f
g(Z) -> 1/(1+math:exp(-Z)).

% handle messages and update state record accordingly
update({layer_before, LayerBefore}, State) ->
	State#neuron{layer_before = LayerBefore};
update({layer_after, LayerAfter}, State) ->
	case State#neuron.type of
		output ->
			Thetas = undefined;
		_Else ->
            % generate truly random numbers
            <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
            random:seed(A, B, C),
            BiasTheta = random_init_bias_theta(),
			Thetas = random_init_thetas(LayerAfter)
	end,
	State#neuron{layer_after = LayerAfter, thetas = Thetas, bias_theta = BiasTheta};
update({forwardprop, X}, State) ->
	% collect inputs from layer before
	NewInputs = [X | State#neuron.inputs],
	case length(NewInputs) =:= length(State#neuron.layer_before) of
		true -> % if we have all the inputs, calculate activation and send to next layer
			case State#neuron.type of
				input -> Activation = lists:sum(NewInputs);
				_Else -> Activation	= g(lists:sum(NewInputs)+State#neuron.bias_theta)
			end,
			lists:map(fun(Pid) ->
						gen_server:cast(Pid, {forwardprop, maps:get(Pid, State#neuron.thetas)*Activation})
					  end,
					  State#neuron.layer_after),
			State#neuron{inputs = [], activation = Activation};
		false -> % update inputs collected
			State#neuron{inputs = NewInputs}
	end;
update({backprop, NextPid, D}, State) ->
	case State#neuron.type of
		output -> % get difference from actual class and send to previous layer
			Delta = State#neuron.activation - D,
			lists:map(fun(Pid) -> gen_server:cast(Pid, {backprop, self(), Delta}) end, State#neuron.layer_before);
		_Else ->
			% collect deltas from layer after
			NewDeltas = maps:put(NextPid, D, State#neuron.deltas),
			case maps:size(NewDeltas) =:= length(State#neuron.layer_after) of
				true -> % if we have all the delta terms
					case State#neuron.type of
						input -> % tell network we have finished training on this instance
							lists:map(fun(Pid) -> gen_server:cast(Pid, finished) end, State#neuron.layer_before);
						hidden -> % compute Delta and send to previous layer
							Delta = (State#neuron.activation*(1-State#neuron.activation))*
									(lists:sum(lists:map(fun(Pid) ->
															maps:get(Pid, State#neuron.thetas)*maps:get(Pid, State#neuron.deltas)
									  					end,
									  					State#neuron.layer_after))+State#neuron.bias_theta,
							lists:map(fun(Pid) -> gen_server:cast(Pid, {backprop, self(), Delta}) end, State#neuron.layer_before),
							State#neuron{delta = State#neuron.delta + Delta, deltas = maps:new()}
					end;
				false -> % update deltas collected
					State#neuron{deltas = NewDeltas}
			end
	end;
update(_Msg, State) ->
    State.

% returns map of random initial theta value for each Pid in LayerBefore as a key
random_init_thetas(Pids) ->
    lists:foldr(fun(Pid, AccumMap) ->
                    maps:put(Pid, (random:uniform()*(2.0*?INIT_EPSILON))-?INIT_EPSILON, AccumMap)
                end,
                maps:new(),
                Pids).

random_init_bias_theta() -> (random:uniform()*(2.0*?INIT_EPSILON))-?INIT_EPSILON.
