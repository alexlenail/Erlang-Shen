-module (shen_neuron).
-export ([start/1]).


-define(INIT_EPSILON, 0.0001).


% LayerBefore and LayerAfter are PID lists. 
start(M) ->

	receive
		{Pid, LayerBefore, LayerAfter} -> ok
	end,

	% random initialization of Thetas
	ThetaMap = maps:new(),
	lists:map(fun(Pid) -> maps:put(Pid, (random:uniform()*(2.0*?INIT_EPSILON))-?INIT_EPSILON, ThetaMap) end, LayerBefore),

	Accumulator = maps:new(),
	lists:map(fun(Pid) -> maps:put(Pid, 0, Accumulator) end, LayerAfter),

	% loop runs M times
	% returns Accumulator

	% use Accumulator to update ThetaMap

	% do again. 

	% send messages to first layer. 
	% receive from last layer. 
	% send actual to last layer. 
	% make sure backprop stops for first layer. 

	loop(LayerBefore, LayerAfter, ThetaMap, maps:new(), maps:new(), Accumulator).

	% Bigloop -> 
	% 	loop 
	% 	Bigloop()



loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator) -> 
	receive
		{Pid, Activation} when lists:member(Pid, LayerBefore) ->
			maps:put(Pid, Activation, ActivationMap),
			if maps:size() =:= length(LayerBefore) ->
				forward(LayerBefore, LayerAfter, ActivationMap, ThetaMap),
				loop(LayerBefore, LayerAfter, ThetaMap, maps:new(), DeltaMap), Accumulator;
			true ->
				loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator)
			end;
		{Pid, Delta} when member(Pid, LayerAfter) ->
			maps:put(Pid, Delta, DeltaMap),
			if maps:size(DeltaMap) =:= length(LayerAfter) ->
				NewAccumulator = backprop(LayerBefore, LayerAfter, DeltaMap, ThetaMap, Accumulator),
				loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, maps:new(), NewAccumulator);
			true -> 
				loop(LayerBefore, LayerAfter, ThetaMap, ActivationMap, DeltaMap, Accumulator)
			end
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

forward(LayerBefore, LayerAfter, ActivationMap, ThetaMap) ->
	Activation = g(lists:sum(lists:map(fun(Pid) -> maps:get(Pid, ActivationMap) * maps:get(Pid, ThetaMap) end, LayerBefore))),
	lists:map(fun(Pid) -> Pid ! {self(), Activation} end, LayerAfter),
	Activation. 


g(Z) -> 1/(1+math:exp(-Z)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

backprop(LayerBefore, LayerAfter, DeltaMap, ThetaMap, Accumulator) -> 
	
	Error = lists:sum(lists:map(fun(Pid) -> maps:get(Pid, DeltaMap) * maps:get(Pid, ThetaMap) end, LayerAfter)),
	Delta = Activation * (1- Activation) * Error,

	lists:map(fun(Pid) -> 
				Change = Activation * maps:get(Pid, DeltaMap),
				maps:put(Pid, maps:get(Pid, Accumulator) + Change, Accumulator)
			end,
		LayerAfter),

	lists:map(fun(Pid) -> Pid ! {self(), Delta} end, LayerBefore), 

	Accumulator.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gradient checking?


