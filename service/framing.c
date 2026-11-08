#include "framing.h"
#include <stdlib.h>
#include <string.h>

int service_frame_with_limit(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, unsigned *output, unsigned *capacity)
{
    uint32_t *ids = NULL;
    size_t count = 0;
    char error[512];
    if (!lunelette_tokenizer_encode_with_limit(tokenizer, text, length, input_limit, &ids, &count,
                                    error, sizeof(error))) {
        return 0;
    }
    unsigned structural = initial ? 3 : 2;
    if (count + structural > *capacity) {
        free(ids);
        return 0;
    }
    unsigned used = 0;
    if (initial) {
        output[used++] = LUNELETTE_TOKEN_BOS;
    }
    output[used++] = LUNELETTE_TOKEN_INST;
    for (size_t i = 0; i < count; i++) {
        output[used++] = ids[i];
    }
    output[used++] = LUNELETTE_TOKEN_END_INST;
    *capacity = used;
    free(ids);
    return 1;
}

const char *service_calculator_v13_declaration(void)
{
    return "[{\"type\": \"function\", \"function\": {\"name\": \"calculator\", "
        "\"description\": \"Calculate with two bounded integers.\", \"parameters\": {"
        "\"type\": \"object\", \"properties\": {\"operation\": {\"type\": \"string\", "
        "\"enum\": [\"add\", \"subtract\", \"multiply\"]}, \"a\": {\"type\": \"integer\", "
        "\"minimum\": -1000000, \"maximum\": 1000000}, \"b\": {\"type\": \"integer\", "
        "\"minimum\": -1000000, \"maximum\": 1000000}}, \"required\": [\"operation\", \"a\", "
        "\"b\"], \"additionalProperties\": false}}}]";
}

const char *service_investigation_v13_declaration(void)
{
    return "[{\"type\": \"function\", \"function\": {\"name\": \"calculator\", "
        "\"description\": \"Calculate with two bounded integers.\", \"parameters\": {"
        "\"type\": \"object\", \"properties\": {\"operation\": {\"type\": \"string\", "
        "\"enum\": [\"add\", \"subtract\", \"multiply\"]}, \"a\": {\"type\": \"integer\", "
        "\"minimum\": -1000000, \"maximum\": 1000000}, \"b\": {\"type\": \"integer\", "
        "\"minimum\": -1000000, \"maximum\": 1000000}}, \"required\": [\"operation\", \"a\", "
        "\"b\"], \"additionalProperties\": false}}}, {\"type\": \"function\", \"function\": {"
        "\"name\": \"lookup\", \"description\": \"Look up a frozen stock document.\", "
        "\"parameters\": {\"type\": \"object\", \"properties\": {\"document_id\": {"
        "\"type\": \"string\", \"enum\": [\"larch-stock\", \"cedar-stock\"]}}, "
        "\"required\": [\"document_id\"], \"additionalProperties\": false}}}]";
}

const char *service_observation_v13_declaration(void)
{
    return "[{\"type\": \"function\", \"function\": {\"name\": \"lookup\", "
        "\"description\": \"Look up a saved synthetic observation, not a live sensor.\", "
        "\"parameters\": {\"type\": \"object\", \"properties\": {\"document_id\": {"
        "\"type\": \"string\", \"enum\": [\"pump-readings\"]}}, "
        "\"required\": [\"document_id\"], \"additionalProperties\": false}}}]";
}

static int frame_profile(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result, unsigned profile,
    unsigned *output, unsigned *capacity);

int service_frame_observation(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity)
{
    return frame_profile(tokenizer, text, length, input_limit, initial, result, 3,
                         output, capacity);
}

int service_frame_tools(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity)
{
    return frame_profile(tokenizer, text, length, input_limit, initial, result, 1,
                         output, capacity);
}

int service_frame_investigation(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result,
    unsigned *output, unsigned *capacity)
{
    return frame_profile(tokenizer, text, length, input_limit, initial, result, 2,
                         output, capacity);
}

static int frame_profile(const lunelette_tokenizer *tokenizer, const unsigned char *text,
    size_t length, size_t input_limit, int initial, int result, unsigned profile,
    unsigned *output, unsigned *capacity)
{
    uint32_t *ids = NULL, *declaration = NULL;
    size_t count = 0, tools = 0;
    char error[512];
    const char *json = profile == 3 ? service_observation_v13_declaration() :
        profile == 2 ? service_investigation_v13_declaration() :
        service_calculator_v13_declaration();
    if (!lunelette_tokenizer_encode_with_limit(tokenizer, text, length, input_limit,
            &ids, &count, error, sizeof(error))) {
        return 0;
    }
    if (initial && !result && !lunelette_tokenizer_encode_with_limit(tokenizer,
            (const unsigned char *)json, strlen(json), 4096, &declaration, &tools,
            error, sizeof(error))) {
        free(ids);
        return 0;
    }
    size_t structural = initial && !result ? 5 : 2;
    if (count + tools + structural > *capacity) {
        free(ids);
        free(declaration);
        return 0;
    }
    unsigned used = 0;
    if (initial && !result) {
        output[used++] = 1;
        output[used++] = 5;
        for (size_t i = 0; i < tools; i++) {
            output[used++] = declaration[i];
        }
        output[used++] = 6;
    }
    output[used++] = result ? 7 : 3;
    for (size_t i = 0; i < count; i++) {
        output[used++] = ids[i];
    }
    output[used++] = result ? 8 : 4;
    *capacity = used;
    free(ids);
    free(declaration);
    return 1;
}
