SHELL := /bin/bash
.DEFAULT_GOAL := compile-skip-deps # this is detected by SublimeText3 as default action to recompile

REBAR3 := $(shell which rebar3)
CURRENT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: precompile
precompile:
	echo "precompiling, rebar3:" $(REBAR3)

.PHONY: shell-dev
shell-dev: precompile
	$(REBAR3) as dev shell

.PHONY: shell-prod
shell-prod: precompile
	$(REBAR3) as prod shell

.PHONY: release
release: precompile
	rm -rf $(CURRENT_DIR)/_build/prod/rel/shen
	$(REBAR3) as prod tar
	$(eval VSN=$(shell cat $(CURRENT_DIR)/src/shen.app.src | grep vsn | perl -pe 's/^.*vsn[^"]+//g' | sgrep -d '"\"".."\""' | sed -e 's/"//g'))
	rm -rf $(CURRENT_DIR)/temprelease
	mkdir $(CURRENT_DIR)/temprelease
	tar zxf $(CURRENT_DIR)/_build/prod/rel/shen/shen-$(VSN).tar.gz -C $(CURRENT_DIR)/temprelease

.PHONY: xref
xref:
	$(REBAR3) xref | egrep -v unused

.PHONY: nifs
nifs:
	echo "no nifs"

.PHONY: compile-skip-deps
compile-skip-deps: precompile
	$(REBAR3) as dev compile skip_deps=true ## this is the default Makefile recipe which is intercepted by SublimeText3 as BUILD action

.PHONY: compile
compile: precompile
	$(REBAR3) as dev compile

.PHONY: clean
clean:
	rm -rf $(CURRENT_DIR)/_build

.PHONY: dialyzer
dialyzer: precompile
	$(REBAR3) as dev dialyzer	

.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# **** dialyzer ****
.dialyzer_generic.plt:
	dialyzer					\
		--build_plt				\
		--output_plt .dialyzer_generic.plt	\
		--apps erts kernel stdlib compiler sasl os_mon mnesia \
			tools public_key crypto ssl

.dialyzer_sockjs.plt: .dialyzer_generic.plt
	dialyzer				\
		--no_native			\
		--add_to_plt			\
		--plt .dialyzer_generic.plt	\
		--output_plt .dialyzer_sockjs.plt -r deps/*/ebin

distclean::
	rm -f .dialyzer_sockjs.plt

dialyze-old: .dialyzer_sockjs.plt
	@dialyzer	 		\
	  --plt .dialyzer_sockjs.plt	\
	  --no_native			\
	  --fullpath			\
		-Wrace_conditions	\
		-Werror_handling	\
		-Wunmatched_returns	\
	  ebin



