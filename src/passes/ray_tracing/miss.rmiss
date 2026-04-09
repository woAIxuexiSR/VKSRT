#version 460
#extension GL_EXT_ray_tracing : enable

struct HitInfo
{
    bool isHit;
    vec3 color;
};

layout(location = 0) rayPayloadInEXT HitInfo hitInfo;

void main()
{
    hitInfo.isHit = false;
    hitInfo.color = vec3(0.2);
}
