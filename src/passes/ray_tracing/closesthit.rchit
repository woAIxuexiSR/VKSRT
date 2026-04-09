#version 460
#extension GL_EXT_ray_tracing : enable

layout(binding = 3) buffer VertexBuffer { float vertices[]; };
layout(binding = 4) buffer IndexBuffer { uint indices[]; };
layout(binding = 5) buffer MaterialBuffer { vec4 materials[]; };

struct HitInfo
{
    bool isHit;
    vec3 color;
};

layout(location = 0) rayPayloadInEXT HitInfo hitInfo;

hitAttributeEXT vec3 attribs;

layout(shaderRecordEXT) buffer HitGroupSBT
{
    int matIndex;
    int vertexOffset;
    int indexOffset;
};

void main()
{
    vec4 matColor = materials[matIndex];

    hitInfo.isHit = true;
    hitInfo.color = matColor.rgb;
}
