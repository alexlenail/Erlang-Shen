REBAR := ./rebar

.PHONY: all clean

all:
	$(REBAR) compile

clean:
	$(REBAR) clean
