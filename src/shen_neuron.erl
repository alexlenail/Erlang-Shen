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
				 inputs = [], activation, thetas, delta = 0, deltas}).

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
			Thetas = random_init_thetas(LayerAfter)
	end,
	State#neuron{layer_after = LayerAfter, thetas = Thetas};
update({forwardprop, X}, State) ->
	% collect inputs from layer before
	NewInputs = [X | State#neuron.inputs],
	case length(NewInputs) =:= length(State#neuron.layer_before) of
		true -> % if we have all the inputs, calculate activation and send to next layer
			case State#neuron.type of
				input -> Activation = lists:sum(NewInputs);
				_Else -> Activation	= g(lists:sum(NewInputs))
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
		output ->
			Delta = State#neuron.activation - D,
			lists:map(fun(Pid) -> gen_server:cast(Pid, {backprop, self(), Delta}) end, State#neuron.layer_before),
		_Else ->
			% collect deltas from layer after
			NewDeltas = maps:put(NextPid, D, State#neuron.deltas),
			case maps:size(NewDeltas) =:= length(State#neuron.layer_after) of
				true -> % if we have all the delta terms, calculate Delta term and send to previous layer
					case State#neuron.type of
						input -> % tell network we have finished training on this instance
							lists:map(fun(Pid) -> gen_server:cast(Pid, finished), State#neuron.layer_before);
						hidden -> % 
							Delta = (State#neuron.activation*(1-State#neuron.activation))*
									lists:sum(lists:map(fun(Pid) ->
															maps:get(Pid, State#neuron.thetas)*maps:get(Pid, State#neuron.deltas)
									  					end,
									  					State#neuron.layer_after)),
							lists:map(fun(Pid) -> gen_server:cast(Pid, {backprop, self(), Delta}) end, State#neuron.layer_before),
							State#neuron{delta = State#neuron.delta + Delta, deltas = maps:new()};
				false -> % update deltas collected
					State#neuron{deltas = NewDeltas}
			end
	end;
update(_Msg, State) ->
    State.

% returns map of random initial theta value for each Pid in LayerBefore as a key
random_init_thetas(LayerBefore) -> 
    lists:foldr(fun(Pid, AccumMap) ->
                    % generate a truly random number
                    <<A:32, B:32, C:32>> = crypto:rand_bytes(12),
                    random:seed(A, B, C),
                    maps:put(Pid, (random:uniform()*(2.0*?INIT_EPSILON))-?INIT_EPSILON, AccumMap)
                end,
                maps:new(),
                LayerBefore).




% outerLoop(LayerBefore, LayerAfter, ThetaMap, M) -> ok.

	% % Initialize the Accumulator, accumulates error
	% Accumulator = maps:new(),
	% lists:map(fun(Pid) -> maps:put(Pid, 0, Accumulator) end, LayerAfter),

	% % One iteration of training
	% Accumulated = loop(LayerBefore, LayerAfter, ThetaMap, maps:new(), maps:new(), Accumulator, M),

	% % Compute Partial Derivatives
	% DMap = maps:new(),
	% lists:map(fun(Pid) -> maps:put(Pid, (1/M) * maps:get(Pid, Accumulated) + Lambda * maps:get(Pid, ThetaMap), DMap) end, LayerAfter),
	% maps:put(Bias, (1/M) * maps:get(Pid, Accumulated), DMap),

	% % Update Weights
	% NewThetaMap = maps:new(),
	% lists:map(fun(Pid) -> maps:put(Pid, maps:get(Pid, ThetaMap) - Alpha * maps:get(Pid, DMap), NewThetaMap) end, LayerBefore),
	% maps:put(Bias, maps:get(Bias, ThetaMap) - Alpha * maps:get(Bias, DMap), NewThetaMap),

	% outerLoop(NewThetaMap).

	% send messages to first layer. 
	% receive from last layer. 
	% send actual to last layer. 
	% make sure backprop stops for first layer. 


% loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator, M) -> ok.
	% receive
	% 	{Pid, Data} ->
	% 		case lists:member(Pid, LayerBefore) of
	% 			true -> member_layer;
	% 			false ->
	% 				case lists:member(Pid, LayerAfter) of
	% 					true -> member_layerafter;
	% 					false -> error
	% 				end
	% 		end
	% 	{Pid, input, Input}
	% end.



	% ****************** Needed to put in case statements because we can't evaluate expressions in if *************
	% Might want to move the logic below into separate functions for forward and backprop


	% 	{Pid, Activation} when lists:member(Pid, LayerBefore) ->
	% 		maps:put(Pid, Activation, ActivationMap),
	% 		if maps:size() =:= length(LayerBefore) ->
	% 			forward(LayerBefore, LayerAfter, ActivationMap, ThetaMap),
	% 			loop(LayerBefore, LayerAfter, ThetaMap, maps:new(), DeltaMap), Accumulator, M;
	% 		true ->
	% 			loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator, M)
	% 		end;
	% 	{Pid, Delta} when member(Pid, LayerAfter) ->
	% 		maps:put(Pid, Delta, DeltaMap),
	% 		case maps:size(DeltaMap) of
	% 			length(LayerAfter) ->
	% 				NewAccumulator = backprop(LayerBefore, LayerAfter, DeltaMap, ThetaMap, Accumulator),
	% 				if M > 1 ->
	% 					loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, maps:new(), NewAccumulator, M-1);
	% 				M =:= 1 -> NewAccumulator
	% 				end;
	% 			_ -> loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator, M)
	% 		end
	% end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% forward(LayerBefore, LayerAfter, ActivationMap, ThetaMap) -> ok.
	% LinearCombination = lists:sum(lists:map(fun(Pid) -> maps:get(Pid, ActivationMap) * maps:get(Pid, ThetaMap) end, LayerBefore)),
	% Activation = g(LinearCombination + maps:get(Bias, ThetaMap)),
	% lists:map(fun(Pid) -> Pid ! {self(), Activation} end, LayerAfter),
	% Activation. 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% backprop(LayerBefore, LayerAfter, DeltaMap, ThetaMap, Accumulator) -> ok.
	
	% Error = lists:sum(lists:map(fun(Pid) -> maps:get(Pid, DeltaMap) * maps:get(Pid, ThetaMap) end, LayerAfter)),
	% Delta = Activation * (1- Activation) * Error,

	% lists:map(fun(Pid) -> 
	% 			Change = Activation * maps:get(Pid, DeltaMap),
	% 			maps:put(Pid, maps:get(Pid, Accumulator) + Change, Accumulator)
	% 		end,
	% 	LayerAfter),

	% lists:map(fun(Pid) -> Pid ! {self(), Delta} end, LayerBefore), 

	% Accumulator.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gradient checking?
