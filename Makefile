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
$(BUILD):
	mkdir -p $@

$(BUILD)/test_generation: tests/test_generation.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_generation.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)
$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_tokenizer.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)
$(BUILD)/test_chat: tests/test_chat.c src/chat.c include/chat.h $(TOKENIZER_HEADERS) include/runtime_limits.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_chat.c src/chat.c -o $@ $(LDLIBS)

$(BUILD)/test_safetensors: tests/test_safetensors.c tests/fp8_golden.h src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_safetensors.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_embeddings: tests/compare_embeddings.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_embeddings.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_weights: tests/test_metal_weights.c src/metal_weights.c src/metal_weights.h include/metal.h src/model_facts.c include/model_facts.h src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_weights.c src/metal_weights.c src/model_facts.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_zero_work: tests/test_metal_zero_work.c $(CLI_SOURCES) $(CLI_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_zero_work.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_metal_memory_guard: tests/test_metal_memory_guard.c src/metal_memory_guard.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

core-test: $(BUILD)/test_metal_memory_guard $(BUILD)/test_generation $(BUILD)/test_tokenizer $(BUILD)/test_chat $(BUILD)/test_safetensors $(BUILD)/compare_embeddings $(BUILD)/test_metal_weights $(BUILD)/test_metal_zero_work
	./$(BUILD)/test_safetensors $(BUILD)/test-safetensors.tmp $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors $(BUILD)/synthetic-mismatch.safetensors
	./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors
	@if ./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-mismatch.safetensors >$(BUILD)/mismatch.log 2>&1; then cat $(BUILD)/mismatch.log; exit 1; fi
	grep 'mismatch token position 0 column 0: expected 3f800001, got 3f800000' $(BUILD)/mismatch.log
	cd $(BUILD) && ./test_tokenizer
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
METAL_HEADERS = src/metal_memory_guard.h include/safetensors.h include/metal.h src/metal_internal.h src/metal_mps.h src/metal_weights.h $(MODEL_HEADERS)
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

$(BUILD)/test_metal: tests/test_metal.m tests/rope_golden.h $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) tests/test_metal.m $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

.PHONY: metal-test
$(BUILD)/test_metal_compact: tests/test_metal_compact.m $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) tests/test_metal_compact.m $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

metal-test: $(BUILD)/test_metal $(BUILD)/compare_metal $(BUILD)/test_metal_compact
	./$(BUILD)/test_metal
	@./$(BUILD)/test_metal_compact; status=$$?; if [ $$status -eq 77 ]; then echo "SKIP compact model-free GPU integration"; elif [ $$status -ne 0 ]; then exit $$status; fi
	./$(BUILD)/compare_metal --self-test

$(BUILD)/metal_snapshot.o: tests/metal_snapshot.m $(METAL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) -c $< -o $@

$(BUILD)/compare_metal: $(BUILD)/metal_snapshot.o tests/compare_metal.c $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
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
$(BUILD)/test_native: tests/compare_native.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DNATIVE_OFFLINE $(LDFLAGS) $< -o $@ $(LDLIBS)
core-test: native-self-test
.PHONY: native-self-test
native-self-test: $(BUILD)/test_native
	./$(BUILD)/test_native --self-test

ifeq ($(shell uname -s),Darwin)
$(BUILD)/compare_native: tests/compare_native.c $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(METAL_TARGET) $(LDFLAGS) $< $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)
.PHONY: native-regression
native-regression: $(BUILD)/compare_native
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" -a -n "$(BASELINE)" -a -n "$(MODE)" || { echo 'SKIP: explicit WEIGHTS FIXTURES BASELINE MODE required (not a pass)'; exit 77; }
	./$(BUILD)/compare_native check "$(WEIGHTS)" "$(MODE)" "$(BASELINE)" $(foreach n,6 24 150 248,"$(FIXTURES)/prefill$(n).ids" "$(FIXTURES)/prefill$(n).safetensors")
endif

# Model-free POSIX host foundation; never links the production inference runtime.
SERVICE_SOURCES = service/session.c src/tokenizer_json.c
SERVICE_HEADERS = service/session.h src/tokenizer_json.h
SERVICE_COMMIT = $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
SERVICE_DIRTY = $(shell test -z "$$(git status --porcelain 2>/dev/null)" && echo false || echo true)
$(BUILD)/test_service_server: service/http.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"$(SERVICE_COMMIT)"' -DSERVICE_DIRTY='"$(SERVICE_DIRTY)"' $(LDFLAGS) service/http.c $(SERVICE_SOURCES) -o $@ -pthread
$(BUILD)/test_service_session: tests/test_service_session.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_session.c $(SERVICE_SOURCES) -o $@ -pthread
$(BUILD)/test_service_transport: tests/test_service_transport.c service/http.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_transport.c $(SERVICE_SOURCES) -o $@ -pthread
.PHONY: service-protocol-test service-test
core-test: service-protocol-test
service-protocol-test: $(BUILD)/test_service_session $(BUILD)/test_service_pieces $(BUILD)/test_service_metrics
	./$(BUILD)/test_service_session
	./$(BUILD)/test_service_pieces
	./$(BUILD)/test_service_metrics
# Python is optional test/client tooling, not a native runtime dependency.
service-test: service-protocol-test $(BUILD)/test_service_server $(BUILD)/test_service_transport
	./$(BUILD)/test_service_transport
	python3 tests/test_service_http.py ./$(BUILD)/test_service_server

ifeq ($(shell uname -s),Darwin)
$(BUILD)/lunelette-server: service/http.c service/native.c service/native.h service/framing.c service/framing.h $(SERVICE_SOURCES) $(SERVICE_HEADERS) $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_NATIVE -DSERVICE_COMMIT='"$(SERVICE_COMMIT)"' -DSERVICE_DIRTY='"$(SERVICE_DIRTY)"' $(METAL_TARGET) $(LDFLAGS) service/http.c service/native.c service/framing.c service/session.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ -pthread $(LDLIBS) $(METAL_LIBS)
endif

$(BUILD)/test_service_framing: tests/test_service_framing.c service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) $(LDFLAGS) tests/test_service_framing.c service/framing.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_service_pieces: tests/test_service_pieces.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_pieces.c $(SERVICE_SOURCES) -o $@ -pthread

ifeq ($(shell uname -s),Darwin)
# Optional CPU-only native adapter test: runtime symbols are explicit test stubs.
$(BUILD)/test_service_native: tests/test_service_native.c service/native.c service/native.h include/metal.h service/framing.c service/framing.h $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -Wno-deprecated-declarations -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_native.c service/native.c service/framing.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES) -o $@ $(LDLIBS)
endif

$(BUILD)/test_service_metrics: tests/test_service_metrics.c $(SERVICE_SOURCES) $(SERVICE_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc -Iservice $(CFLAGS) -DSERVICE_COMMIT='"test"' -DSERVICE_DIRTY='"true"' $(LDFLAGS) tests/test_service_metrics.c src/tokenizer_json.c -o $@ -pthread
