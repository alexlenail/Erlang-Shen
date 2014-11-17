-module(shen_parse).
-export([get_sdfsd/1]).


% we can handle 2 types of data sets, arff and NIST standardized ocr data

get_data({arff, TrainSet, TestSet}) -> ok;
get_data({ocr_nist, TrainSet, TestSet}) -> ok;
get_data(_) -> {error, invalid_data}.
