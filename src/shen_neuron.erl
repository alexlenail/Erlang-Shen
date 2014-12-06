-module(shen_neuron).
-behaviour(gen_server).

%% API
-export([start_link/1]).

%% Gen Server Callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
         terminate/2, code_change/3]).

-define(NETWORK, shen_network).
-define(INIT_EPSILON, 0.0001).
-define(LAMBDA, 0.0005).
-define(ALPHA, 0.1).


%% ===================================================================
%% API Functions
%% ===================================================================

start_link(Args) ->
    gen_server:start_link(?MODULE, Args, []).


%% ===================================================================
%% Gen Server Callbacks
%% ===================================================================

-record(neuron, {type, layer_before, layer_after, inputs = [],
                 activation, thetas, deltas, delta_collector}).

init({neuron_type, Type}) ->
	% io:format("init neuron (~w)~n", [self()]),
    % generate truly random numbers
    <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
    random:seed(A, B, C),
	InitState = #neuron{type = Type, deltas = maps:new(), delta_collector = maps:new()},
    {ok, InitState}.

handle_cast(Msg, State) ->
	NewState = update(Msg, State),
    % erlang:display(NewState),
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

% sigmoid function
g(Z) -> 1 / (1 + math:exp(-Z)).

% handle messages and update state record accordingly
update({layer_before, LayerBefore}, State) ->
    % erlang:display({self(), layer_before, LayerBefore}),
	State#neuron{layer_before = LayerBefore};
update({layer_after, LayerAfter}, State) ->
    % erlang:display({self(), layer_after, LayerAfter}),
	case State#neuron.type of
		output ->
			Thetas = undefined;
		_Else ->
			Thetas = random_init_thetas(LayerAfter)
	end,
	State#neuron{layer_after = LayerAfter, thetas = Thetas};
update({forwardprop, _PrevPid, X}, State) ->
    % erlang:display({forwardprop, _PrevPid, X}),
	% collect inputs from layer before
	NewInputs = [X | State#neuron.inputs],
	case length(NewInputs) =:= length(State#neuron.layer_before) of
		true -> % if we have all the inputs, calculate activation and send to next layer
			case State#neuron.type of
				input -> Activation = lists:sum(NewInputs);
				_Else1 ->
                    Activation = g(lists:sum(NewInputs))
			end,
			lists:map(fun(Pid) ->
						case State#neuron.type of 
							output -> ?NETWORK ! {forwardprop, self(), Activation};
							_Else2 -> gen_server:cast(Pid, {forwardprop, self(), maps:get(Pid, State#neuron.thetas) * Activation})
						end
					end,
					State#neuron.layer_after),
			State#neuron{inputs = [], activation = Activation};
		false -> % update inputs collected
			State#neuron{inputs = NewInputs}
	end;
update({backprop, NextPid, D}, State) ->
    erlang:display({backprop, NextPid, D}),
	case State#neuron.type of
		output -> % get difference from actual class and send to previous layer
			Delta = State#neuron.activation - D,
			lists:map(fun(Pid) ->
                        % send back delta
                        gen_server:cast(Pid, {backprop, self(), Delta})
                      end,
                      State#neuron.layer_before),
            State;
		_Else ->
			% collect deltas from layer after
			NewDeltaCollector = maps:put(NextPid, D, State#neuron.delta_collector),
			case maps:size(NewDeltaCollector) =:= length(State#neuron.layer_after) of
				true -> % if we have all the delta terms from the next layer
                    % update uppercase delta terms (accumulated delta)
                    BigDeltas = lists:foldl(fun(Pid, AccumMap) ->
                                                Update = State#neuron.activation * maps:get(Pid, NewDeltaCollector),
                                                case maps:find(Pid, AccumMap) of 
                                                    {ok, V} -> maps:update(Pid, V + Update, AccumMap);
                                                    error -> maps:put(Pid, Update, AccumMap)
                                                end
                                            end,
                                            State#neuron.deltas,
                                            State#neuron.layer_after),
                    case State#neuron.type of
						input -> % tell network we have finished training on this instance
							?NETWORK ! finished_instance,
                            State#neuron{deltas = BigDeltas, delta_collector = maps:new()};
						hidden -> % compute Delta and send to previous layer
							Delta = (State#neuron.activation * (1 - State#neuron.activation))*
									lists:sum(lists:map(fun(Pid) ->
															maps:get(Pid, State#neuron.thetas) * maps:get(Pid, NewDeltaCollector)
									  					end,
									  					State#neuron.layer_after)),
							lists:map(fun(Pid) -> gen_server:cast(Pid, {backprop, self(), Delta}) end, State#neuron.layer_before),							
							State#neuron{deltas = BigDeltas, delta_collector = maps:new()}
					end;
				false -> % update deltas collected
					State#neuron{delta_collector = NewDeltaCollector}
			end
	end;
update({descend_gradient, M}, State) ->
    erlang:display({descend_gradient, self(), State#neuron.type, State#neuron.deltas, State#neuron.thetas}),
	Dij = lists:foldl(fun(Pid, Map) ->
			            maps:put(Pid, (1/M) * maps:get(Pid, State#neuron.deltas) + 
                                      ?LAMBDA * maps:get(Pid, State#neuron.thetas), Map)
		              end,
		              maps:new(),
		              State#neuron.layer_after),
	NewThetas = lists:foldl(fun(Pid, Map) -> 
			maps:put(Pid, maps:get(Pid, State#neuron.thetas) -
                     ?ALPHA * maps:get(Pid, Dij), Map)
		end,
		maps:new(),
		State#neuron.layer_after),
	?NETWORK ! finished_descent,
	State#neuron{thetas = NewThetas, deltas = maps:new()};
update(_Msg, State) ->
    State.

% returns map of random initial theta value for each Pid in LayerBefore as a key
random_init_thetas(Pids) ->
    lists:foldr(fun(Pid, AccumMap) ->
                    maps:put(Pid, (random:uniform()*(2.0 * ?INIT_EPSILON)) - ?INIT_EPSILON, AccumMap)
                end,
                maps:new(),
                Pids).
