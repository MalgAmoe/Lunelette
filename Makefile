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

test: all $(BUILD)/test_math
	./$(BUILD)/test_math
	./$(BUILD)/lunelette --version
	./$(BUILD)/lunelette --help
	@if ./$(BUILD)/lunelette --invalid >/dev/null 2>&1; then exit 1; fi

sanitize:
	$(MAKE) BUILD=build/sanitize CFLAGS='-std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' test

clean:
	rm -rf build
