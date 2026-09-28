struct Matrix {
    vertex: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct ColorBlock {
    color: vec3<f32>,
    opacity: f32,
}
@group(1) @binding(0) var<uniform> material: ColorBlock;

struct Global {
    projection: mat4x4<f32>,
}
@group(2) @binding(0) var<uniform> global: Global;

@group(3) @binding(0) var surface_texture: texture_2d<f32>;
@group(3) @binding(1) var surface_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32,
           @location(0) pos: vec2<f32>,
           @location(1) texcoord: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = global.projection * matrix[instance_index].vertex * vec4<f32>(pos, 0.0, 1.0);
    out.texcoord = texcoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(surface_texture, surface_sampler, in.texcoord);
    return texel * vec4<f32>(material.color, material.opacity);
}
