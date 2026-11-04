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

$(BUILD)/lunelette: $(CLI_SOURCES) $(CLI_HEADERS) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(CLI_SOURCES) -o $@ $(LDLIBS)
