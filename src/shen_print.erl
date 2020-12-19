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
    lager:info("========================================~n"),
    lager:info(Text, Format),
    lager:info("========================================~n"),
    ok.

event(Text, Format) ->
    lager:info(Text, Format),
    ok.
