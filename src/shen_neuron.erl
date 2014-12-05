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

% maybe different records for different types
-record(neuron, {network_pid, type, layer_before, layer_after,
				 thetas}).

init({{network_pid, NetworkPid}, {neuron_type, Type}}) ->
	io:format("init neuron (~w)~n", [self()]),
	InitState = #neuron{network_pid = NetworkPid, type = Type},
	erlang:display(InitState),
    {ok, InitState}.

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

% handle messages and update state record accordingly
update({layer_before, LayerBefore}, State) ->
	case State#neuron.type of
		input -> Thetas = undefined;
		_Else ->
			Thetas = random_init_thetas(LayerBefore)
	end,
	State#neuron{layer_before = LayerBefore, thetas = Thetas};
update({layer_after, LayerAfter}, State) ->
	State#neuron{layer_after = LayerAfter};
update({forwardprop, X}, State) ->
	% case
	State;
update({backprop, X}, State) ->
	State;
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


% g(Z) -> 1/(1+math:exp(-Z)).

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
