-module (network).
-export ([function/arity]).


% init (num hidden layers, num nodes per layer (or just a list of numbers of nodes if we want to vary nodes per layer))
    % for each layer, start correct number of nodes and save pids
        % pass previous layer pids to spawning node as param
        % send list of pids of completed layer to previous layer
    % potentially return a pid to our "main" that we can send data to

% learn
    % send data to input layer


% bias units?


