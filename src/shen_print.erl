%% ===================================================================
%% shen_print.erl
%%
%% Helper module for printing progress information for the user.
%%
%% ===================================================================

-module(shen_print).

%% API
-export([title/2, event/2]).


%% ===================================================================
%% API Functions
%% ===================================================================

title(Text, Format) ->
    io:format("========================================~n"),
    io:format(Text, Format),
    io:format("========================================~n"),
    ok.

event(Text, Format) ->
    io:format(Text, Format),
    ok.
