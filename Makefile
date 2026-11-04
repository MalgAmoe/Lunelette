MODEL_SOURCES = src/model_facts.c src/cpu_weights.c
MODEL_HEADERS = include/model_facts.h include/runtime_limits.h src/cpu_weights.h

CC = cc
CPPFLAGS += -Iinclude
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS += -lm
BUILD = build

.PHONY: all test sanitize clean
all: $(BUILD)/lunelette

$(BUILD):
	mkdir -p $@

$(BUILD)/test_math: tests/test_math.c src/math.c include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_math.c src/math.c -o $@ $(LDLIBS)

$(BUILD)/test_kernels: tests/test_kernels.c src/math.c include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_kernels.c src/math.c -o $@ $(LDLIBS)

test: $(BUILD)/compare_generation $(BUILD)/test_generation $(BUILD)/test_forward $(BUILD)/compare_forward $(BUILD)/compare_block $(BUILD)/test_attention $(BUILD)/compare_attention $(BUILD)/compare_projections all $(BUILD)/test_math $(BUILD)/test_kernels $(BUILD)/test_safetensors $(BUILD)/compare_embeddings
	./$(BUILD)/test_safetensors $(BUILD)/test-safetensors.tmp $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors $(BUILD)/synthetic-mismatch.safetensors
	./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors
	@if ./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-mismatch.safetensors >$(BUILD)/mismatch.log 2>&1; then cat $(BUILD)/mismatch.log; exit 1; fi
	grep 'mismatch token position 0 column 0: expected 3f800001, got 3f800000' $(BUILD)/mismatch.log
	./$(BUILD)/compare_projections --self-test
	./$(BUILD)/test_attention
	./$(BUILD)/compare_attention --self-test
	./$(BUILD)/compare_block --self-test
	./$(BUILD)/compare_forward --self-test
	./$(BUILD)/test_forward
	cd $(BUILD) && ./test_tokenizer
	cd $(BUILD) && ./test_chat
	cd $(BUILD) && sh $(abspath tests/test_cli.sh) ./lunelette
	./$(BUILD)/test_generation
	./$(BUILD)/test_cache
	./$(BUILD)/compare_cache --self-test
	./$(BUILD)/compare_generation --self-test
	./$(BUILD)/test_math
	./$(BUILD)/test_kernels
	./$(BUILD)/lunelette --version
	./$(BUILD)/lunelette --help
	@if ./$(BUILD)/lunelette --invalid >/dev/null 2>&1; then exit 1; fi

sanitize:
	$(MAKE) BUILD=build/sanitize CFLAGS='-std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' test

clean:
	rm -rf build

$(BUILD)/test_safetensors: tests/test_safetensors.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_safetensors.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_embeddings: tests/compare_embeddings.c src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_embeddings.c src/safetensors.c -o $@ $(LDLIBS)

.PHONY: integration
integration: $(BUILD)/compare_embeddings
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=embedding-shard FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_embeddings "$(WEIGHTS)" $(foreach f,short multilingual code chat longer,"$(FIXTURES)/$(f).safetensors")

$(BUILD)/compare_projections: tests/compare_projections.c src/safetensors.c src/math.c include/safetensors.h include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_projections.c src/safetensors.c src/math.c -o $@ $(LDLIBS)

$(BUILD)/test_attention: tests/test_attention.c tests/rope_golden.h src/attention.c include/attention.h $(MODEL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_attention.c src/attention.c -o $@ $(LDLIBS)

$(BUILD)/compare_attention: tests/compare_attention.c src/attention.c src/safetensors.c include/attention.h $(MODEL_HEADERS) include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_attention.c src/attention.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_block: tests/compare_block.c src/block.c $(MODEL_SOURCES) src/forward_internal.h $(MODEL_HEADERS) src/attention.c src/math.c src/safetensors.c include/block.h include/attention.h include/lunelette.h include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_block.c src/block.c $(MODEL_SOURCES) src/attention.c src/math.c src/safetensors.c -o $@ $(LDLIBS)

.PHONY: block-integration
block-integration: $(BUILD)/compare_block
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=layer0-shard FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_block "$(WEIGHTS)" $(foreach f,short multilingual code chat longer,"$(FIXTURES)/$(f).safetensors")

$(BUILD)/compare_forward: tests/compare_forward.c tests/full_depth.h src/forward.c src/block.c $(MODEL_SOURCES) src/forward_internal.h $(MODEL_HEADERS) src/attention.c src/math.c src/safetensors.c include/forward.h include/block.h include/attention.h include/lunelette.h include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_forward.c src/forward.c src/block.c $(MODEL_SOURCES) src/attention.c src/math.c src/safetensors.c -o $@ $(LDLIBS)

.PHONY: forward-integration
forward-integration: $(BUILD)/compare_forward
	@test -n "$(WEIGHTS)" -a -n "$(WEIGHTS2)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=shard1 WEIGHTS2=shard2 FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_forward "$(WEIGHTS)" "$(WEIGHTS2)" $(foreach f,short multilingual code chat longer,"$(FIXTURES)/$(f).safetensors")

$(BUILD)/test_forward: tests/test_forward.c src/forward.c src/block.c $(MODEL_SOURCES) src/forward_internal.h $(MODEL_HEADERS) src/attention.c src/math.c src/safetensors.c include/forward.h include/block.h include/attention.h include/lunelette.h include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_forward.c src/forward.c src/block.c $(MODEL_SOURCES) src/attention.c src/math.c src/safetensors.c -o $@ $(LDLIBS)

GENERATION_SOURCES = src/cache.c src/generation.c src/forward.c src/block.c $(MODEL_SOURCES) src/attention.c src/math.c src/safetensors.c
GENERATION_HEADERS = $(MODEL_HEADERS) include/cache.h include/generation.h include/forward.h include/block.h include/attention.h include/lunelette.h include/safetensors.h src/forward_internal.h

$(BUILD)/test_generation: tests/test_generation.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_generation.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/compare_generation: tests/compare_generation.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_generation.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_cache: tests/test_cache.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/cache.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_cache.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)


$(BUILD)/compare_cache: tests/compare_cache.c $(GENERATION_SOURCES) $(GENERATION_HEADERS) include/cache.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_cache.c $(GENERATION_SOURCES) -o $@ $(LDLIBS)

test: $(BUILD)/test_cache $(BUILD)/compare_cache

TOKENIZER_SOURCES = src/tokenizer.c src/tokenizer_json.c
TOKENIZER_HEADERS = include/tokenizer.h src/tokenizer_json.h src/tokenizer_unicode.h
CLI_SOURCES = src/main.c src/chat.c $(TOKENIZER_SOURCES) $(GENERATION_SOURCES)
CLI_HEADERS = include/chat.h $(TOKENIZER_HEADERS) $(GENERATION_HEADERS)

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(TOKENIZER_SOURCES) $(TOKENIZER_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_tokenizer.c $(TOKENIZER_SOURCES) -o $@ $(LDLIBS)

$(BUILD)/test_chat: tests/test_chat.c src/chat.c include/chat.h $(TOKENIZER_HEADERS) $(GENERATION_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_chat.c src/chat.c -o $@ $(LDLIBS)

test: $(BUILD)/test_tokenizer $(BUILD)/test_chat

# Compare the saved mode each invocation. Mark a changed config phony so even
# same-timestamp switches rebuild the CLI (BSD make timestamp resolution).
BACKEND_CONFIG = METAL=$(if $(filter 1,$(METAL)),1,0)
ifneq ($(shell cat $(BUILD)/backend.config 2>/dev/null),$(BACKEND_CONFIG))
.PHONY: $(BUILD)/backend.config
endif
$(BUILD)/backend.config: | $(BUILD)
	@printf '%s\n' '$(BACKEND_CONFIG)' > $@.tmp
	@mv $@.tmp $@

$(BUILD)/lunelette: $(BUILD)/backend.config $(CLI_SOURCES) $(CLI_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(METAL_CPPFLAGS) $(METAL_TARGET) $(CLI_SOURCES) $(METAL_OBJECTS) -o $@ $(LDLIBS) $(METAL_LIBS)

# Metal is opt-in. The portable C11 targets never see Apple headers/frameworks.
ifeq ($(METAL),1)
ifneq ($(shell uname -s),Darwin)
$(error METAL=1 requires macOS and Apple Silicon; omit METAL=1 for portable CPU)
endif
METAL_C_SOURCES = src/metal_weights.c src/metal_yarn.c
METAL_OBJC_SOURCES = src/metal_runtime.m src/metal_encode.m
METAL_OBJECTS = $(patsubst src/%.c,$(BUILD)/%.o,$(METAL_C_SOURCES)) $(patsubst src/%.m,$(BUILD)/%.o,$(METAL_OBJC_SOURCES))
METAL_HEADERS = include/metal.h src/metal_internal.h src/metal_weights.h $(MODEL_HEADERS)
METAL_CPPFLAGS = -DLUNELETTE_USE_METAL
METAL_TARGET = -mmacosx-version-min=15.0
METAL_LIBS = -framework Foundation -framework Metal
METAL_FLAGS = -fobjc-arc -fno-objc-exceptions -ffp-contract=off -mmacosx-version-min=15.0

$(BUILD)/embed_shader: tools/embed_shader.c | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic $< -o $@

$(BUILD)/metal_shader.h: src/metal_kernels.metal $(BUILD)/embed_shader
	./$(BUILD)/embed_shader $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/metal_%.o: src/metal_%.c $(METAL_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=off -mmacosx-version-min=15.0 -c $< -o $@

$(BUILD)/metal_%.o: src/metal_%.m $(METAL_HEADERS) $(BUILD)/metal_shader.h | $(BUILD)
	$(CC) $(CPPFLAGS) -I$(BUILD) $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) -c $< -o $@

$(BUILD)/lunelette: $(METAL_OBJECTS) include/metal.h
$(BUILD)/test_metal: tests/test_metal.m tests/rope_golden.h $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) -Isrc $(filter-out -std=c11,$(CFLAGS)) $(METAL_FLAGS) $(LDFLAGS) tests/test_metal.m $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

.PHONY: metal-test
metal-test: $(BUILD)/test_metal $(BUILD)/compare_metal
	./$(BUILD)/test_metal
	./$(BUILD)/compare_metal --self-test

test: metal-test
$(BUILD)/compare_metal: tests/compare_metal.c $(METAL_OBJECTS) $(METAL_HEADERS) src/model_facts.c src/safetensors.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -mmacosx-version-min=15.0 $(LDFLAGS) tests/compare_metal.c $(METAL_OBJECTS) src/model_facts.c src/safetensors.c -o $@ $(LDLIBS) $(METAL_LIBS)

endif

# Metadata preflight is portable and model-free even when Metal is not enabled.
$(BUILD)/test_metal_weights: tests/test_metal_weights.c src/metal_weights.c src/metal_weights.h src/model_facts.c include/model_facts.h src/safetensors.c include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) $(LDFLAGS) tests/test_metal_weights.c src/metal_weights.c src/model_facts.c src/safetensors.c -o $@ $(LDLIBS)

.PHONY: metal-weights-test
metal-weights-test: $(BUILD)/test_metal_weights
	./$(BUILD)/test_metal_weights

test: metal-weights-test
