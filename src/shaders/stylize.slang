// Shared stylization function module
// Applies nonlinear function g to radiance based on material style type
module stylize;

import material;
// Style type constants defined in material.slang:
// STYLE_NONE=0, STYLE_TOON=1, STYLE_EIGHT_BIT=2, STYLE_SATURATION=3, STYLE_HALFTONE=4, STYLE_PALETTE=5

// --- Color space conversion helpers ---

float linearToSrgb1(float c)
{
    if (c <= 0.0031308)
        return 12.92 * c;
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float srgbToLinear1(float c)
{
    if (c <= 0.04045)
        return c / 12.92;
    return pow((c + 0.055) / 1.055, 2.4);
}

float3 linearToSrgb(float3 c)
{
    return float3(linearToSrgb1(c.r), linearToSrgb1(c.g), linearToSrgb1(c.b));
}

float3 srgbToLinear(float3 c)
{
    return float3(srgbToLinear1(c.r), srgbToLinear1(c.g), srgbToLinear1(c.b));
}

float3 rgbToHsv(float3 c)
{
    float cmax = max(c.r, max(c.g, c.b));
    float cmin = min(c.r, min(c.g, c.b));
    float delta = cmax - cmin;

    float h = 0.0;
    if (delta > 1e-6)
    {
        if (cmax == c.r)
            h = (c.g - c.b) / delta;
        else if (cmax == c.g)
            h = 2.0 + (c.b - c.r) / delta;
        else
            h = 4.0 + (c.r - c.g) / delta;
        h /= 6.0;
        if (h < 0.0) h += 1.0;
    }

    float s = (cmax > 1e-6) ? (delta / cmax) : 0.0;
    float v = cmax;
    return float3(h, s, v);
}

float3 hsvToRgb(float3 hsv)
{
    float h = hsv.x * 6.0;
    float s = hsv.y;
    float v = hsv.z;

    float c = v * s;
    float x = c * (1.0 - abs(fmod(h, 2.0) - 1.0));
    float m = v - c;

    float3 rgb;
    if (h < 1.0)      rgb = float3(c, x, 0.0);
    else if (h < 2.0) rgb = float3(x, c, 0.0);
    else if (h < 3.0) rgb = float3(0.0, c, x);
    else if (h < 4.0) rgb = float3(0.0, x, c);
    else if (h < 5.0) rgb = float3(x, 0.0, c);
    else               rgb = float3(c, 0.0, x);

    return rgb + float3(m, m, m);
}

// --- Stylize functions ---

public float3 applyStylize(float3 L, int styleType, float styleParam0)
{
    if (styleType == STYLE_NONE)
        return L;

    if (styleType == STYLE_TOON)
    {
        // Luminance quantization preserving hue
        float bandWidth = styleParam0; // e.g. 0.2
        if (bandWidth <= 0.0) return L;
        float lum = (L.r + L.g + L.b) / 3.0;
        if (lum <= 0.0) return L;
        float3 hue = L / lum;
        if (lum > 1.0)
        {
            float newLum = floor(lum / bandWidth) * bandWidth;
            return hue * newLum;
        }
        else
        {
            float srgbLum = linearToSrgb1(lum);
            float newLum = floor(srgbLum / bandWidth) * bandWidth;
            return hue * srgbToLinear1(newLum);
        }
    }

    if (styleType == STYLE_EIGHT_BIT)
    {
        // sRGB space 8-8-4 quantization
        float3 srgb = linearToSrgb(clamp(L, float3(0.0), float3(1.0)));
        srgb.r = round(srgb.r * 8.0) / 8.0;
        srgb.g = round(srgb.g * 8.0) / 8.0;
        srgb.b = round(srgb.b * 4.0) / 4.0;
        return srgbToLinear(srgb);
    }

    if (styleType == STYLE_SATURATION)
    {
        // HSV saturation adjustment: S = S^power
        float3 srgb = linearToSrgb(clamp(L, float3(0.0), float3(1.0)));
        float3 hsv = rgbToHsv(srgb);
        hsv.y = pow(hsv.y, styleParam0);
        return srgbToLinear(hsvToRgb(hsv));
    }

    if (styleType == STYLE_HALFTONE)
    {
        // Simplified halftone: luminance threshold
        float lum = (L.r + L.g + L.b) / 3.0;
        float threshold = styleParam0;
        return (lum > threshold) ? L : L * 0.1;
    }

    if (styleType == STYLE_PALETTE)
    {
        // Simplified palette: quantize to N uniform levels
        float levels = max(styleParam0, 2.0);
        float3 srgb = linearToSrgb(clamp(L, float3(0.0), float3(1.0)));
        srgb = floor(srgb * levels) / levels;
        return srgbToLinear(srgb);
    }

    return L;
}
