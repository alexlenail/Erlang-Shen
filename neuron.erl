-module (neuron).
-export ([init/2]).


% LayerBefore and LayerAfter are PID lists. 
start(LayerBefore) ->
	% random initialization of Theta
	% receive next layer pids
	receive
		{Pid, LayerAfter} -> loop(LayerBefore, LayerAfter, Thetas, maps:new(), maps:new())
	end.


loop(LayerBefore, LayerAfter, Thetas, ActivationMap, DeltaMap) -> 
	receive
		{Pid, Activation} when member(Pid, LayerBefore) ->
			maps:put(Pid, Activation, ActivationMap),
			if maps:size() =:= length(LayerBefore) ->
				forward(ActivationMap, Thetas),
				loop(LayerBefore, LayerAfter, Thetas, maps:new(), DeltaMap);
			true ->
				loop(LayerBefore, LayerAfter, Thetas, ActivationMap, DeltaMap)
			end
		{Pid, Delta} when member(Pid, LayerAfter) ->
			maps:put(Pid, Delta, DeltaMap),
			if maps:size(DeltaMap) =:= length(LayerAfter) ->
				NewThetas = forward(DeltaMap),
				loop(LayerBefore, LayerAfter, NewThetas, ActivationMap, maps:new());
			true -> 
				loop(LayerBefore, LayerAfter, Thetas, ActivationMap, DeltaMap)
			end
	end.


forward(LayerBefore, LayerAfter, ActivationMap, Thetas) ->
	Activation = g(lists:sum(lists:map(fun(Pid) -> maps:get(Pid, ActivationMap) * maps:get(Pid, ThetaMap) end, LayerBefore))),
	lists:map(fun(Pid) -> Pid ! {self(), Activation}, LayerAfter).


g(Z) -> 1/(1+math:exp(-Z))





=== 
backprop() -> 

compute J(theta)
send that to all nodes in the past. 


=== 
% gradient checking?


