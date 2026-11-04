#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// FP32 Ministral3 formulas adapted from Transformers; see NOTICE.
// One SIMDgroup/output, Kahan lane accumulation followed by the pinned SIMD sum.
kernel void project(device const float *weights [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]])
{
    uint row = group % dims.x;
    uint token = group / dims.x;
    float sum = 0;
    float correction = 0;
    for (uint column = lane; column < dims.y; column += width) {
        float product = weights[row * dims.y + column] * input[token * dims.y + column];
        float value = product - correction;
        float next = sum + value;
        correction = (next - sum) - value;
        sum = next;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[token * dims.x + row] = sum;
    }
}

kernel void norm(device const float *weights [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]])
{
    float sum = 0;
    for (uint column = lane; column < dims.x; column += width) {
        float value = input[token * dims.x + column];
        sum += value * value;
    }
    float scale = rsqrt(simd_sum(sum) / float(dims.x) + 1e-5f);
    for (uint column = lane; column < dims.x; column += width) {
        output[token * dims.x + column] = weights[column] *
            (input[token * dims.x + column] * scale);
    }
}

// modes: residual add, SiLU, gated multiply.
kernel void element(device const float *first [[buffer(0)]],
    device const float *second [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], uint index [[thread_position_in_grid]])
{
    if (index >= dims.x) {
        return;
    }
    float value = first[index];
    output[index] = dims.y == 0 ? value + second[index] :
        (dims.y == 1 ? value / (1.0f + exp(-value)) : value * second[index]);
}

kernel void rope(device const float *input [[buffer(0)]],
    device const float *cosine [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], device const float *sine [[buffer(4)]],
    device const uint *positions [[buffer(5)]], uint index [[thread_position_in_grid]])
{
    if (index >= dims.x * dims.y) {
        return;
    }
    uint token = index / dims.x;
    uint column = index % 128;
    float rotated = column < 64 ? -input[index + 64] : input[index - 64];
    float value = input[index] * cosine[token * 128 + column] +
        rotated * sine[token * 128 + column];
    if (dims.z) {
        // Integer context block count is the checkpoint's query scaling rule.
        value *= 1.0f + 0.1f * log(1.0f + float(positions[token] / 16384));
    }
    output[index] = value;
}

// One SIMDgroup/query/head; causal softmax scratch [count,32,prefix+count].
kernel void scores(device const float *query [[buffer(0)]],
    device const float *keys [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]])
{
    uint token = group / 32;
    uint head = group % 32;
    uint stride = dims.x + dims.w;
    float maximum = -INFINITY;
    for (uint position = 0; position <= token + dims.w; position++) {
        float sum = 0;
        for (uint column = lane; column < 128; column += width) {
            sum += query[token * 4096 + head * 128 + column] *
                keys[position * 1024 + (head / 4) * 128 + column];
        }
        sum = simd_sum(sum) * 0.08838834764831845f;
        if (lane == 0) {
            output[group * stride + position] = sum;
        }
        maximum = max(maximum, sum);
    }
    simdgroup_barrier(mem_flags::mem_device);
    float total = 0;
    for (uint position = lane; position <= token + dims.w; position += width) {
        float value = exp(output[group * stride + position] - maximum);
        output[group * stride + position] = value;
        total += value;
    }
    total = simd_sum(total);
    for (uint position = lane; position < stride; position += width) {
        output[group * stride + position] = position <= token + dims.w ?
            output[group * stride + position] / total : 0;
    }
}

kernel void attend(device const float *probabilities [[buffer(0)]],
    device const float *values [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]], uint index [[thread_position_in_grid]])
{
    if (index >= dims.x * 4096) {
        return;
    }
    uint token = index / 4096;
    uint head = (index % 4096) / 128;
    uint column = index % 128;
    float sum = 0;
    for (uint position = 0; position <= token + dims.w; position++) {
        sum += probabilities[(token * 32 + head) * (dims.x + dims.w) + position] *
            values[position * 1024 + (head / 4) * 128 + column];
    }
    output[index] = sum;
}
