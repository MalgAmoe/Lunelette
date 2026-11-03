CC = cc
CPPFLAGS += -Iinclude
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS += -lm
BUILD = build

.PHONY: all test sanitize clean
all: $(BUILD)/lunelette

$(BUILD):
	mkdir -p $@

$(BUILD)/lunelette: src/main.c include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) src/main.c -o $@ $(LDLIBS)

$(BUILD)/test_math: tests/test_math.c src/math.c include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_math.c src/math.c -o $@ $(LDLIBS)

$(BUILD)/test_kernels: tests/test_kernels.c src/math.c include/lunelette.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_kernels.c src/math.c -o $@ $(LDLIBS)

test: $(BUILD)/compare_block $(BUILD)/test_attention $(BUILD)/compare_attention $(BUILD)/compare_projections all $(BUILD)/test_math $(BUILD)/test_kernels $(BUILD)/test_safetensors $(BUILD)/compare_embeddings
	./$(BUILD)/test_safetensors $(BUILD)/test-safetensors.tmp $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors $(BUILD)/synthetic-mismatch.safetensors
	./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-match.safetensors
	@if ./$(BUILD)/compare_embeddings $(BUILD)/synthetic-weights.safetensors $(BUILD)/synthetic-mismatch.safetensors >$(BUILD)/mismatch.log 2>&1; then cat $(BUILD)/mismatch.log; exit 1; fi
	grep 'mismatch token position 0 column 0: expected 3f800001, got 3f800000' $(BUILD)/mismatch.log
	./$(BUILD)/compare_projections --self-test
	./$(BUILD)/test_attention
	./$(BUILD)/compare_attention --self-test
	./$(BUILD)/compare_block --self-test
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

$(BUILD)/test_attention: tests/test_attention.c tests/rope_golden.h src/attention.c include/attention.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/test_attention.c src/attention.c -o $@ $(LDLIBS)

$(BUILD)/compare_attention: tests/compare_attention.c src/attention.c src/safetensors.c include/attention.h include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_attention.c src/attention.c src/safetensors.c -o $@ $(LDLIBS)

$(BUILD)/compare_block: tests/compare_block.c src/block.c src/attention.c src/math.c src/safetensors.c include/block.h include/attention.h include/lunelette.h include/safetensors.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) tests/compare_block.c src/block.c src/attention.c src/math.c src/safetensors.c -o $@ $(LDLIBS)

.PHONY: block-integration
block-integration: $(BUILD)/compare_block
	@test -n "$(WEIGHTS)" -a -n "$(FIXTURES)" || { echo 'Set WEIGHTS=layer0-shard FIXTURES=fixture-directory'; exit 1; }
	./$(BUILD)/compare_block "$(WEIGHTS)" $(foreach f,short multilingual code chat longer,"$(FIXTURES)/$(f).safetensors")
