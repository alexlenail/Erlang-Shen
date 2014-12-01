-module(shen_parse).

%% API
-export([arff/1]).


%% ===================================================================
%% ARFF API and Internal Functions
%% ===================================================================

arff(Fname) ->
    case file:open(Fname, [read, raw]) of
        {error, Reason} -> erlang:display(Reason), {error, Reason};
        {ok, Fd} -> arff_line_by_line(Fd, 0, [], [])
    end.


%% ===================================================================
%% ARFF Internal Functions
%% ===================================================================

arff_line_by_line(Fd, NumAttrs, Classes, Instances) ->
    case file:read_line(Fd) of
        {error, Reason} -> {error, Reason};
        eof ->
            file:close(Fd),
            {ok, {NumAttrs, Classes, Instances}};
        {ok, Data} ->
            L = string:to_lower(string:strip(Data)),
            Line = string:substr(L, 1, string:len(L)-1),
            case arff_line_type(Line) of
                skipline -> arff_line_by_line(Fd, NumAttrs, Classes, Instances);
                {classes, UnsplitClasses} -> arff_line_by_line(Fd, NumAttrs, arff_get_classes(UnsplitClasses), Instances);
                attribute -> arff_line_by_line(Fd, NumAttrs+1, Classes, Instances);
                instance -> arff_line_by_line(Fd, NumAttrs, Classes, [arff_get_instance(Line) | Instances])
            end
    end.

arff_line_type("") -> skipline;
arff_line_type("@data") -> skipline;
arff_line_type(Line) ->
    case string:chr(Line, $%) of
        1 -> skipline;
        _ ->
            case string:str(Line, "@relation") of
                1 -> skipline;
                _ ->
                    case string:str(Line, "@attribute") of
                        1 ->
                            SplitLine = string:tokens(Line, " "),
                            case lists:nth(2, SplitLine) of
                                "class" -> {classes, lists:nth(3,SplitLine)};
                                _ -> attribute
                            end;
                        _ -> instance
                    end
            end
    end.

arff_get_classes(UnsplitClasses) ->
    strlist_cast_numeric(string:tokens(string:substr(UnsplitClasses, 2, string:len(UnsplitClasses)-2), ",")).

arff_get_instance(Line) ->
    strlist_cast_numeric(string:tokens(Line, ",")).


%% ===================================================================
%% General Internal Functions
%% ===================================================================

strlist_cast_numeric(List) -> lists:map(fun(X) -> str_try_cast_numeric(X) end, List).

str_try_cast_numeric(N) ->
    case string:to_float(N) of
        {error, no_float} ->
            case string:to_integer(N) of
                {error, no_integer} -> N;
                {I, _Rest} -> I
            end;
        {F, _Rest} -> F
    end.
