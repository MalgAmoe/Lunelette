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

$(BUILD)/test_safetensors: tests/test_safetensors.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_safetensors.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_embeddings: tests/compare_embeddings.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_embeddings.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_weights: tests/test_metal_weights.c src/metal_weights.c src/metal_weights.h src/model_facts.c include/model_facts.h src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_weights.c src/metal_weights.c src/model_facts.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/test_metal_zero_work: tests/test_metal_zero_work.c $(CLI_SOURCES) $(CLI_HEADERS) include/metal.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_zero_work.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

core-test: $(BUILD)/test_generation $(BUILD)/test_tokenizer $(BUILD)/test_chat $(BUILD)/test_safetensors $(BUILD)/compare_embeddings $(BUILD)/test_metal_weights $(BUILD)/test_metal_zero_work
	./$(BUILD)/test_safetensors $(BUILD)/test-safetensors.tmp $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors $(BUILD)/synthetic-mismatch.safetensors
	./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors
	@if ./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-mismatch.safetensors >$(BUILD)/mismatch.log 2>&1; then cat $(BUILD)/mismatch.log; exit 1; fi
	grep 'mismatch token position 0 column 0: expected 3f800001, got 3f800000' $(BUILD)/mismatch.log
	cd $(BUILD) && ./test_tokenizer
	cd $(BUILD) && ./test_chat
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
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=embedding-shard FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_embeddings "$(WEIGHTS)" $(foreach f,short multilingual code chat longer,"$(FIXTURES)/$(f).safetensors")

# Portable core-test exercises host formats and policy, never inference.
ifeq ($(shell uname -s),Darwin)
METAL_C_SOURCES = src/metal_weights.c src/metal_yarn.c
METAL_OBJC_SOURCES = src/metal_runtime.m src/metal_encode.m src/metal_mps.m
METAL_OBJECTS = $(patsubst src/%.c,$(BUILD)/%.o,$(METAL_C_SOURCES)) $(patsubst src/%.m,$(BUILD)/%.o,$(METAL_OBJC_SOURCES))
METAL_HEADERS = include/safetensors.h include/metal.h src/metal_internal.h src/metal_mps.h src/metal_weights.h $(MODEL_HEADERS)
METAL_TARGET = -mmacosx-version-min=15.0
METAL_LIBS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
METAL_FLAGS = -fobjc-arc -fno-objc-exceptions -ffp-contract=off -mmacosx-version-min=15.0

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
metal-test: $(BUILD)/test_metal $(BUILD)/compare_metal
	./$(BUILD)/test_metal
	./$(BUILD)/compare_metal --self-test

$(BUILD)/compare_metal: tests/compare_metal.c $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -mmacosx-version-min=15.0 $(LDFLAGS) tests/compare_metal.c $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

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
