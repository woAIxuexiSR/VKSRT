// Orthonormal Basis utility for local/world coordinate transforms
module onb;

public struct ONB
{
    public float3 u; // tangent
    public float3 v; // bitangent
    public float3 w; // normal (z-axis in local space)
};

// Build ONB from a surface normal
public ONB buildONB(float3 n)
{
    ONB basis;
    basis.w = normalize(n);
    float3 a = (abs(basis.w.x) > 0.9) ? float3(0, 1, 0) : float3(1, 0, 0);
    basis.v = normalize(cross(basis.w, a));
    basis.u = cross(basis.w, basis.v);
    return basis;
}

// Transform direction from local (z-up) to world space
public float3 localToWorld(ONB basis, float3 d)
{
    return d.x * basis.u + d.y * basis.v + d.z * basis.w;
}

// Transform direction from world to local (z-up) space
public float3 worldToLocal(ONB basis, float3 d)
{
    return float3(dot(d, basis.u), dot(d, basis.v), dot(d, basis.w));
}
