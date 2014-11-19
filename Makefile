REBAR := ./rebar

.PHONY: all clean

all:
	$(REBAR) get-deps && $(REBAR) compile

# doc:
# 	$(REBAR) doc

# test:
# 	$(REBAR) eunit

clean:
	$(REBAR) clean && $(REBAR) delete-deps

# release: all test
# 	typer -r ./src/ > /dev/null