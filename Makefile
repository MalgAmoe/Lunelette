CC = cc
CPPFLAGS += -Iinclude
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS += -lm
BUILD = build
MODEL_HEADERS = include/model_facts.h include/runtime_limits.h
GENERATION_SOURCES = src/generation.c
GENERATION_HEADERS = include/generation.h $(MODEL_HEADERS)
TOKENIZER_SOURCES = src/tokenizer.c src/tokenizer_json.c
TOKENIZER_HEADERS = include/tokenizer.h src/tokenizer_json.h src/tokenizer_unicode.h
CLI_SOURCES = src/main.c src/chat.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) src/model_facts.c src/safetensors.c
CLI_HEADERS = include/lunelette.h include/chat.h $(TOKENIZER_HEADERS) $(GENERATION_HEADERS) include/safetensors.h src/metal_weights.h

.PHONY: all test core-test sanitize clean integration
all: $(BUILD)/lunelette

# Explicit convenience entrypoints; existing defaults and suites stay unchanged.
.PHONY: server host-test gpu-test python-test interactive-test entrypoint-test dev-check
server: $(BUILD)/lunelette-server
host-test: core-test
gpu-test: metal-test
PYTHON ?= python3
python-test:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*workload*.py'
	$(PYTHON) -B -m unittest discover -s tests -p 'test_replay_logits.py'
	$(PYTHON) -B -m unittest discover -s tests -p 'test_fp8_lookup.py'
interactive-test: $(BUILD)/test_service_server
	$(PYTHON) -B tests/test_interactive_workload.py ./$(BUILD)/test_service_server
entrypoint-test:
	sh tests/test_entrypoints.sh
# Optional full local workflow: GPU + stdlib Python + loopback protocol checks.
# Separate recursive invocations keep suite ordering explicit, including under -j.
dev-check:
	$(MAKE) BUILD="$(BUILD)" server
	$(MAKE) BUILD="$(BUILD)" test
	$(MAKE) BUILD="$(BUILD)" python-test
	$(MAKE) BUILD="$(BUILD)" interactive-test
	$(MAKE) BUILD="$(BUILD)" service-test
	$(MAKE) BUILD="$(BUILD)" shader-facts-test dependency-test entrypoint-test
$(BUILD):
	mkdir -p $@

$(BUILD)/test_generation: tests/test_generation.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_generation.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)
$(BUILD)/test_generation_alloc: tests/test_generation_alloc.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_generation_alloc.c -o $@ $(LDLIBS)
$(BUILD)/test_tokenizer_alloc: tests/test_tokenizer_alloc.c tests/test_tokenizer.c $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_tokenizer_alloc.c src/tokenizer_json.c -o $@ $(LDLIBS)
$(BUILD)/test_tokenizer_bpe: tests/test_tokenizer_bpe.c $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_tokenizer_bpe.c src/tokenizer_json.c -o $@ $(LDLIBS)

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_tokenizer.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)
$(BUILD)/test_chat: tests/test_chat.c src/chat.c include/chat.h $(TOKENIZER_HEADERS) include/runtime_limits.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_chat.c src/chat.c -o $@ $(LDLIBS)

$(BUILD)/test_safetensors: tests/test_safetensors.c tests/fp8_golden.h src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_safetensors.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_embeddings: tests/compare_embeddings.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_embeddings.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_weights: tests/test_metal_weights.c src/metal_weights.c src/metal_weights.h src/metal_plan.h include/metal.h src/model_facts.c $(MODEL_HEADERS) src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_weights.c src/metal_weights.c src/model_facts.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_zero_work: tests/test_metal_zero_work.c $(CLI_SOURCES) $(CLI_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_zero_work.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_metal_memory_guard: tests/test_metal_memory_guard.c src/metal_memory_guard.h src/metal_plan.h include/metal.h $(MODEL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

core-test: $(BUILD)/test_generation_alloc $(BUILD)/test_tokenizer_alloc $(BUILD)/test_tokenizer_bpe $(BUILD)/test_metal_memory_guard $(BUILD)/test_generation $(BUILD)/test_tokenizer $(BUILD)/test_chat $(BUILD)/test_safetensors $(BUILD)/compare_embeddings $(BUILD)/test_metal_weights $(BUILD)/test_metal_zero_work
	./$(BUILD)/test_safetensors $(BUILD)/test-safetensors.tmp $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors $(BUILD)/synthetic-mismatch.safetensors
	./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors
	@if ./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-mismatch.safetensors >$(BUILD)/mismatch.log 2>&1; then cat $(BUILD)/mismatch.log; exit 1; fi
	grep 'mismatch token position 0 column 0: expected 3f800001, got 3f800000' $(BUILD)/mismatch.log
	cd $(BUILD) && ./test_tokenizer
	cd $(BUILD) && ./test_tokenizer_bpe
	cd $(BUILD) && ./test_tokenizer_alloc
	./$(BUILD)/test_generation_alloc
	cd $(BUILD) && ./test_chat
	./$(BUILD)/test_metal_memory_guard
	./$(BUILD)/test_generation
	./$(BUILD)/test_metal_weights $(BUILD)/test-metal-duplicates.tmp
	./$(BUILD)/test_metal_zero_work

test: core-test all metal-test
	cd $(BUILD) && sh $(abspath tests/test_cli.sh) ./lunelette
	./$(BUILD)/lunelette --version

sanitize:
	$(MAKE) BUILD=build/sanitize CFLAGS='-std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' test
clean:
	rm -rf build
integration: $(BUILD)/compare_embeddings
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=FP8-model.safetensors FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_embeddings "$(WEIGHTS)" $(foreach f,embedding6 embedding24 embedding150 embedding248,"$(FIXTURES)/$(f).safetensors")

# Portable core-test exercises host formats and policy, never inference.
ifeq ($(shell uname -s),Darwin)
METAL_C_SOURCES = src/metal_weights.c src/metal_yarn.c
METAL_OBJC_SOURCES = src/metal_runtime.m src/metal_encode.m src/metal_mps.m
METAL_OBJECTS = $(patsubst src/%.c,$(BUILD)/%.o,$(METAL_C_SOURCES)) $(patsubst src/%.m,$(BUILD)/%.o,$(METAL_OBJC_SOURCES))
METAL_HEADERS = src/metal_options.h src/metal_plan.h src/metal_memory_guard.h include/safetensors.h include/metal.h src/metal_internal.h src/metal_mps.h src/metal_weights.h $(MODEL_HEADERS)
METAL_TARGET = -mmacosx-version-min=15.0
METAL_LIBS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
METAL_FLAGS = -fobjc-arc -fno-objc-exceptions -ffp-contract=off -mmacosx-version-min=15.0

$(BUILD)/compare_fp8: tests/compare_fp8.c src/metal_weights.c src/safetensors.c src/model_facts.c $(METAL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) -Wno-deprecated-declarations $(LDFLAGS) tests/compare_fp8.c src/metal_weights.c src/safetensors.c src/model_facts.c -o $@ $(LDLIBS)

$(BUILD)/embed_shader: tools/embed_shader.c include/model_facts.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic $< -o $@

$(BUILD)/metal_shader.h: src/metal_kernels.metal $(BUILD)/embed_shader
	./$(BUILD)/embed_shader $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/metal_%.o: src/metal_%.c $(METAL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -mmacosx-version-min=15.0 -c $< -o $@

$(BUILD)/metal_%.o: src/metal_%.m $(METAL_HEADERS) $(BUILD)/metal_shader.h | $(BUILD)
	$(CC) $(CPPFLAGS) -I$(BUILD) $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) -c $< -o $@

# Checkpoint allocation failure injection is confined to this test runtime object.
METAL_TEST_OBJECTS = $(filter-out $(BUILD)/metal_runtime.o,$(METAL_OBJECTS)) $(BUILD)/test_metal_runtime.o
$(BUILD)/test_metal_runtime.o: src/metal_runtime.m $(METAL_HEADERS) $(BUILD)/metal_shader.h | $(BUILD)
	$(CC) $(CPPFLAGS) -I$(BUILD) $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) -DLUNELETTE_METAL_CHECKPOINT_ALLOC=lunelette_test_checkpoint_alloc -c $< -o $@

$(BUILD)/test_metal: tests/test_metal.m tests/rope_golden.h $(METAL_TEST_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) tests/test_metal.m $(METAL_TEST_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

.PHONY: metal-test
$(BUILD)/test_metal_compact: tests/test_metal_compact.m $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) tests/test_metal_compact.m $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

metal-test: $(BUILD)/test_metal $(BUILD)/compare_metal $(BUILD)/test_metal_compact
	./$(BUILD)/test_metal
	@./$(BUILD)/test_metal_compact; status=$$?; if [ $$status -eq 77 ]; then echo "SKIP compact model-free GPU integration"; elif [ $$status -ne 0 ]; then exit $$status; fi
	./$(BUILD)/compare_metal --self-test

$(BUILD)/metal_snapshot.o: tests/metal_snapshot.m $(METAL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) -c $< -o $@

$(BUILD)/compare_metal: $(BUILD)/metal_snapshot.o tests/compare_metal.c tests/comparator_strategy.h $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -mmacosx-version-min=15.0 $(LDFLAGS) tests/compare_metal.c $(BUILD)/metal_snapshot.o $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

$(BUILD)/lunelette: $(CLI_SOURCES) $(CLI_HEADERS) $(METAL_OBJECTS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(METAL_TARGET) $(CLI_SOURCES) $(METAL_OBJECTS) -o $@ $(LDLIBS) $(METAL_LIBS)
else
$(BUILD)/lunelette metal-test:
	@echo 'The production runtime requires macOS 15+ and Apple Silicon; use core-test for portable host tests.' >&2
	@exit 1
endif

.PHONY: shader-facts-test dependency-test
shader-facts-test:
	CC='$(CC)' python3 tests/test_shader_facts.py
dependency-test:
	sh tests/test_dependencies.sh

# Model-free DS4-style frozen native arithmetic regression. No fixture generation.
ifeq ($(shell uname -s),Darwin)
$(BUILD)/test_metal_reference: tests/test_metal_reference.m $(BUILD)/metal_shader.h | $(BUILD)
	$(CC) $(CPPFLAGS) -I$(BUILD) $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS) -framework Foundation -framework Metal

.PHONY: metal-reference-test
metal-reference-test: $(BUILD)/test_metal_reference
	./$(BUILD)/test_metal_reference tests/frozen/metal-baseline-v1.metal

metal-test: metal-reference-test
endif

ifeq ($(shell uname -s),Darwin)
.PHONY: metal-norm-diagnostic
metal-norm-diagnostic: $(BUILD)/test_metal_reference
	@test -n "$(INPUT)" -a -n "$(NORM_WEIGHTS)" -a -n "$(OUTPUT)" -a -n "$(COLUMNS)" -a -n "$(ROWS)" || { echo 'Set INPUT=raw-f32 NORM_WEIGHTS=raw-f32 OUTPUT=new-file COLUMNS=n ROWS=n'; exit 1; }
	./$(BUILD)/test_metal_reference --norm "$(INPUT)" "$(NORM_WEIGHTS)" "$(OUTPUT)" "$(COLUMNS)" "$(ROWS)"
endif

# Portable native comparator unit tests; no Metal/Python/model dependency.
$(BUILD)/test_native: tests/compare_native.c tests/comparator_strategy.h include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DNATIVE_OFFLINE $(LDFLAGS) $< -o $@ $(LDLIBS)
$(BUILD)/test_comparator_strategy: tests/test_comparator_strategy.c tests/comparator_strategy.h include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $< -o $@
core-test: native-self-test
.PHONY: native-self-test
native-self-test: $(BUILD)/test_native $(BUILD)/test_comparator_strategy
	./$(BUILD)/test_comparator_strategy
	./$(BUILD)/test_native --self-test

ifeq ($(shell uname -s),Darwin)
$(BUILD)/compare_native: tests/compare_native.c tests/comparator_strategy.h $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(METAL_TARGET) $(LDFLAGS) $< $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)
STRATEGY ?= resident-fp32
.PHONY: native-regression
native-regression: $(BUILD)/compare_native
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" -a -n "$(BASELINE)" -a -n "$(MODE)" || { echo 'SKIP: explicit WEIGHTS FIXTURES BASELINE MODE required (not a pass)'; exit 77; }
	./$(BUILD)/compare_native check "$(WEIGHTS)" "$(MODE)" "$(BASELINE)" $(foreach n,6 24 150 248,"$(FIXTURES)/prefill$(n).ids" "$(FIXTURES)/prefill$(n).safetensors") --strategy="$(STRATEGY)"
endif

# Model-free POSIX host foundation; never links the production inference runtime.
SERVICE_SOURCES = service/session.c src/tokenizer_json.c
SERVICE_HEADERS = service/session.h src/tokenizer_json.h
SERVICE_COMMIT = $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
SERVICE_DIRTY = $(shell test -z "$$(git status --porcelain 2>/dev/null)" && echo false || echo true)
# Recheck on every invocation, but invalidate consumers only on content changes.
.PHONY: service-identity-force build-identity-test
$(BUILD)/service-identity: service-identity-force tools/update_service_identity.sh | $(BUILD)
	sh tools/update_service_identity.sh "$@" '$(SERVICE_COMMIT)' '$(SERVICE_DIRTY)'
$(BUILD)/test_service_server: $(BUILD)/service-identity
build-identity-test:
	sh tests/test_build_identity.sh
core-test: build-identity-test
$(BUILD)/test_service_server: service/http.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"$(SERVICE_COMMIT)"' -DSERVICE_DIRTY='"$(SERVICE_DIRTY)"' $(LDFLAGS) service/http.c $(SERVICE_SOURCES) -o $@ -pthread
$(BUILD)/test_service_session: tests/test_service_session.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_session.c $(SERVICE_SOURCES) -o $@ -pthread
$(BUILD)/test_service_transport: tests/test_service_transport.c service/http.c src/metal_plan.h $(MODEL_HEADERS) include/metal.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_transport.c $(SERVICE_SOURCES) -o $@ -pthread
$(BUILD)/test_service_replay: tests/test_service_replay.c tests/test_service_session.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_replay.c $(SERVICE_SOURCES) -o $@ -pthread

.PHONY: service-protocol-test service-test
core-test: service-protocol-test
service-protocol-test: $(BUILD)/test_service_replay $(BUILD)/test_service_tools $(BUILD)/test_service_session $(BUILD)/test_service_pieces $(BUILD)/test_service_metrics $(BUILD)/test_service_chunks $(BUILD)/test_service_json_limits
	./$(BUILD)/test_service_replay
	./$(BUILD)/test_service_tools
	./$(BUILD)/test_service_session
	./$(BUILD)/test_service_pieces
	./$(BUILD)/test_service_metrics
	./$(BUILD)/test_service_chunks
	./$(BUILD)/test_service_json_limits
# Python is optional test/client tooling, not a native runtime dependency.
service-test: service-protocol-test $(BUILD)/test_service_server $(BUILD)/test_service_transport
	./$(BUILD)/test_service_transport
	python3 tests/test_service_http.py ./$(BUILD)/test_service_server

ifeq ($(shell uname -s),Darwin)
$(BUILD)/lunelette-server: $(BUILD)/service-identity service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_NATIVE -DSERVICE_COMMIT='"$(SERVICE_COMMIT)"' -DSERVICE_DIRTY='"$(SERVICE_DIRTY)"' $(METAL_TARGET) $(LDFLAGS) service/http.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ -pthread $(LDLIBS) $(METAL_LIBS)
endif

$(BUILD)/test_service_framing: tests/test_service_framing.c service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) $(LDFLAGS) tests/test_service_framing.c service/framing.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_service_pieces: tests/test_service_pieces.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_pieces.c $(SERVICE_SOURCES) -o $@ -pthread

ifeq ($(shell uname -s),Darwin)
# Optional CPU-only native adapter test: runtime symbols are explicit test stubs.
$(BUILD)/test_service_native: tests/test_service_native.c service/native.c service/native.h $(SERVICE_HEADERS) include/metal.h service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_native.c service/native.c service/framing.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ $(LDLIBS)
endif

$(BUILD)/test_service_metrics: tests/test_service_metrics.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_metrics.c src/tokenizer_json.c -o $@ -pthread

ifeq ($(shell uname -s),Darwin)
# Optional CPU-only native HTTP integration; tokenizer supplied by the caller.
$(BUILD)/test_service_checkpoint_server: tests/test_service_checkpoint_server.c tests/test_service_native.c service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_checkpoint_server.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ -pthread $(LDLIBS)
.PHONY: service-checkpoint-test
service-checkpoint-test: $(BUILD)/test_service_checkpoint_server
	@test -n "$(TOKENIZER)" || { echo 'Supply TOKENIZER=/path/to/official/tokenizer.json'; exit 1; }
	python3 tests/test_service_checkpoint_http.py ./$(BUILD)/test_service_checkpoint_server "$(TOKENIZER)"

$(BUILD)/test_service_replay_native: tests/test_service_replay_native.c tests/test_service_native.c service/native.c service/native.h service/framing.c $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_replay_native.c service/framing.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ -pthread $(LDLIBS)

.PHONY: workload-replay-test
workload-replay-test: $(BUILD)/test_service_replay_native $(BUILD)/test_service_checkpoint_server
	@test -n "$(TOKENIZER)" || { echo 'Supply TOKENIZER=/path/to/official/tekken.json'; exit 1; }
	./$(BUILD)/test_service_replay_native "$(TOKENIZER)"
	python3 tests/test_workload_replay_http.py ./$(BUILD)/test_service_checkpoint_server "$(TOKENIZER)"

.PHONY: workload-scenarios-test
workload-scenarios-test: $(BUILD)/test_service_checkpoint_server
	@test -n "$(TOKENIZER)" || { echo 'Supply TOKENIZER=/path/to/official/tekken.json'; exit 1; }
	python3 tests/test_workload_scenarios_http.py ./$(BUILD)/test_service_checkpoint_server "$(TOKENIZER)"
endif

ifeq ($(shell uname -s),Darwin)
$(BUILD)/test_service_long_native: tests/test_service_long_native.c tests/test_service_native.c service/native.c service/native.h include/metal.h $(SERVICE_HEADERS) service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_long_native.c service/framing.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ $(LDLIBS)

.PHONY: service-long-test
service-long-test: $(BUILD)/test_service_native $(BUILD)/test_service_long_native $(BUILD)/test_service_checkpoint_server
	@test -n "$(TOKENIZER)" || { echo 'Supply TOKENIZER=/path/to/official/tekken.json'; exit 1; }
	./$(BUILD)/test_service_native "$(TOKENIZER)"
	./$(BUILD)/test_service_long_native "$(TOKENIZER)"
	python3 tests/test_service_long_http.py ./$(BUILD)/test_service_checkpoint_server "$(TOKENIZER)"
endif

$(BUILD)/test_service_chunks: tests/test_service_chunks.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_chunks.c $(SERVICE_SOURCES) -o $@ -pthread

$(BUILD)/test_service_json_limits: tests/test_service_json_limits.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_json_limits.c $(SERVICE_SOURCES) -o $@ -pthread

$(BUILD)/test_service_tools: tests/test_service_tools.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_tools.c src/tokenizer_json.c -o $@ -pthread

$(BUILD)/test_service_tools_framing: tests/test_service_tools_framing.c service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) $(LDFLAGS) tests/test_service_tools_framing.c service/framing.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_service_tools_native: tests/test_service_tools_native.c tests/test_service_native.c service/native.c service/native.h $(SERVICE_HEADERS) include/metal.h service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_tools_native.c service/native.c service/framing.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ $(LDLIBS)

# Optional: caller supplies an existing tokenizer and mistral_common 1.13.1 Python.
SERVICE_TOOLS_PYTHON ?= python3
.PHONY: service-tools-tokenizer-test
service-tools-tokenizer-test: $(BUILD)/test_service_tools_framing $(BUILD)/test_service_tools_native
	@test -n "$(TOKENIZER)" || { echo 'TOKENIZER path required (no downloads)'; exit 1; }
	./$(BUILD)/test_service_tools_native "$(TOKENIZER)"
	$(SERVICE_TOOLS_PYTHON) tests/test_service_tools_fixture.py ./$(BUILD)/test_service_tools_framing "$(TOKENIZER)"

# SCRIPTED native socket/client integration, never inference or model evidence.
ifeq ($(shell uname -s),Darwin)
$(BUILD)/test_service_tools_http: tests/test_service_tools_http.c tests/test_service_native.c service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_tools_http.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ -pthread $(LDLIBS)
service-tools-tokenizer-test: service-tools-http-test
.PHONY: service-tools-http-test
service-tools-http-test: $(BUILD)/test_service_tools_http
	@test -n "$(TOKENIZER)" || { echo 'TOKENIZER path required (no downloads)'; exit 1; }
	$(SERVICE_TOOLS_PYTHON) tests/test_service_tools_http.py ./$(BUILD)/test_service_tools_http "$(TOKENIZER)"
endif

$(BUILD)/test_service_lookup: tests/test_service_lookup.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_lookup.c src/tokenizer_json.c -o $@ -pthread
service-protocol-test: service-lookup-test
.PHONY: service-lookup-test
service-lookup-test: $(BUILD)/test_service_lookup
	./$(BUILD)/test_service_lookup

$(BUILD)/test_service_lookup_framing: tests/test_service_lookup_framing.c service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) $(LDFLAGS) tests/test_service_lookup_framing.c service/framing.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)
.PHONY: service-lookup-tokenizer-test
service-lookup-tokenizer-test: $(BUILD)/test_service_lookup_framing
	@test -n "$(TOKENIZER)" || { echo 'TOKENIZER path required (no downloads)'; exit 1; }
	$(SERVICE_TOOLS_PYTHON) tests/test_service_lookup_fixture.py ./$(BUILD)/test_service_lookup_framing "$(TOKENIZER)"

$(BUILD)/test_service_lookup_http: tests/test_service_lookup_http.c tests/test_service_native.c service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_lookup_http.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ -pthread $(LDLIBS)
service-lookup-tokenizer-test: service-lookup-http-test
.PHONY: service-lookup-http-test
service-lookup-http-test: $(BUILD)/test_service_lookup_http
	$(SERVICE_TOOLS_PYTHON) tests/test_service_lookup_http.py ./$(BUILD)/test_service_lookup_http "$(TOKENIZER)"

$(BUILD)/test_service_observation: tests/test_service_observation.c tests/test_service_lookup.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_observation.c src/tokenizer_json.c -o $@ -pthread
.PHONY: service-observation-test
service-protocol-test: service-observation-test
service-observation-test: $(BUILD)/test_service_observation
	./$(BUILD)/test_service_observation

$(BUILD)/test_service_observation_framing: tests/test_service_observation_framing.c service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) $(LDFLAGS) tests/test_service_observation_framing.c service/framing.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)
$(BUILD)/test_service_observation_http: tests/test_service_observation_http.c tests/test_service_native.c service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_observation_http.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ -pthread $(LDLIBS)
.PHONY: service-observation-tokenizer-test
service-observation-tokenizer-test: $(BUILD)/test_service_observation_framing $(BUILD)/test_service_observation_http
	@test -n "$(TOKENIZER)" || { echo 'TOKENIZER path required (no downloads)'; exit 1; }
	$(SERVICE_TOOLS_PYTHON) tests/test_service_observation_fixture.py ./$(BUILD)/test_service_observation_framing "$(TOKENIZER)"
	$(SERVICE_TOOLS_PYTHON) tests/test_service_observation_http.py ./$(BUILD)/test_service_observation_http "$(TOKENIZER)"

# Constructor option admission is host-only, with no Python/model/GPU prerequisites.
$(BUILD)/test_metal_options: tests/test_metal_options.c src/metal_options.h include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)
core-test: metal-options-test
.PHONY: metal-options-test
metal-options-test: $(BUILD)/test_metal_options
	./$(BUILD)/test_metal_options
