#include <metal_stdlib>
using namespace metal;

vertex float4 vertex_shader(uint vertexID [[vertex_id]]) {
    const float4 vertices[] = {
        float4(-1, -1, 0, 1),
        float4( 3, -1, 0, 1),
        float4(-1,  3, 0, 1)
    };
    return vertices[vertexID];
}

fragment float4 fragment_shader(constant uchar &flash_state [[buffer(0)]]) {
    return float4(flash_state, flash_state, flash_state, 1);
}