-module (curry).
-export ([build/3]).


build(Length, BaseFunction) when Length =:= 0 -> BaseFunction;
build(Length, BaseFunction) when Length > 0 -> build(Length-1, fun(P) -> BaseFunction end).

baseFunction(Pid, Activation) -> put Pid-Activation in the KVS






