#include "tool_grammar.h"
#include <string.h>

static int equals(const tj_node *n, const char *text)
{
    return n->length == strlen(text) && !memcmp(n->text, text, n->length);
}

/* Decimal integers only; canonical results additionally reject -0. */
static int bounded_integer(const unsigned char *text, size_t length, uint64_t bound,
                           int canonical, int64_t *value)
{
    size_t i = 0;
    int negative = length && text[0] == '-';
    if (negative) {
        i++;
    }
    if (i == length || (text[i] == '0' && ((canonical && negative) || length - i != 1))) {
        return 0;
    }
    uint64_t number = 0;
    for (; i < length; i++) {
        unsigned digit = (unsigned)text[i] - '0';
        if (digit > 9 || number > (bound - digit) / 10) {
            return 0;
        }
        number = number * 10 + digit;
    }
    *value = negative ? -(int64_t)number : (int64_t)number;
    return 1;
}

int service_tool_calculator(const tj_document *d, service_tool_calculation *output)
{
    unsigned op = tj_field(d, 1, "operation");
    unsigned a = tj_field(d, 1, "a"), b = tj_field(d, 1, "b");
    int64_t av = 0, bv = 0;
    unsigned operation = op && d->nodes[op].kind == '"' ?
        (equals(&d->nodes[op], "add") ? 1 : equals(&d->nodes[op], "subtract") ? 2 :
         equals(&d->nodes[op], "multiply") ? 3 : 0) : 0;
    int valid = operation && a && b && d->nodes[a].kind == '0' && d->nodes[b].kind == '0' &&
        bounded_integer(d->nodes[a].text, d->nodes[a].length, 1000000, 0, &av) &&
        bounded_integer(d->nodes[b].text, d->nodes[b].length, 1000000, 0, &bv);
    if (valid) {
        output->operation = operation;
        output->a = (int)av;
        output->b = (int)bv;
    }
    return valid;
}

static const char *const lookup_documents[] = {"", "larch-stock", "cedar-stock", "pump-readings"};
static const char *const lookup_passages[] = {
    "", "LARCH-STOCK:P1", "CEDAR-STOCK:P1", "PUMP-READINGS:P1"
};

const char *service_tool_document(unsigned document)
{
    return lookup_documents[document];
}

unsigned service_tool_lookup(const tj_document *d, unsigned profile)
{
    unsigned id = tj_field(d, 1, "document_id");
    unsigned document = id && d->nodes[id].kind == '"' ?
        (equals(&d->nodes[id], lookup_documents[1]) ? 1 :
         equals(&d->nodes[id], lookup_documents[2]) ? 2 :
         equals(&d->nodes[id], lookup_documents[3]) ? 3 : 0) : 0;
    if (!((profile == 2 && document >= 1 && document <= 2) ||
          (profile == 3 && document == 3))) {
        document = 0;
    }
    return document;
}

int service_tool_lookup_result(const tj_document *d, unsigned document)
{
    unsigned id = tj_field(d, 1, "document_id");
    unsigned passage = tj_field(d, 1, "passage_id");
    unsigned text = tj_field(d, 1, "text");
    int valid = document >= 1 && document <= 3 && id && passage && text &&
        d->nodes[id].kind == '"' && d->nodes[passage].kind == '"' &&
        d->nodes[text].kind == '"' && d->nodes[text].length && d->nodes[text].length <= 1024 &&
        equals(&d->nodes[id], lookup_documents[document]) &&
        equals(&d->nodes[passage], lookup_passages[document]);
    return valid;
}

int service_tool_calculator_result(const unsigned char *text, size_t length, int64_t *value)
{
    return bounded_integer(text, length, 1000000000000ULL, 1, value);
}
