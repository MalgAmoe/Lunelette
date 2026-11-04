#ifndef LUNELETTE_METAL_MPS_H
#define LUNELETTE_METAL_MPS_H

#include "metal_internal.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

/* Private concrete cache layout, shared only with maintained model-free tests. */
@interface LuneletteMPSCache : NSObject {
@public
    size_t count;
    MPSMatrix *inputs[LUNELETTE_METAL_WEIGHT_COUNT];
    MPSMatrix *weights[LUNELETTE_METAL_WEIGHT_COUNT];
    MPSMatrix *outputs[LUNELETTE_METAL_WEIGHT_COUNT];
    MPSMatrixMultiplication *kernels[7];
}
@end

MPSMatrix *lunelette_metal_mps_view(id<MTLBuffer> buffer, size_t rows, size_t columns);
MPSMatrixMultiplication *lunelette_metal_mps_kernel(id<MTLDevice> device,
    size_t count, size_t rows, size_t columns);
#endif
