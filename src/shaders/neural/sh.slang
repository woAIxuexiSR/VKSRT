module sh;

public static const int MAX_SH_DEGREE = 4;
public static const int MAX_SH_COEFFS = 25;

public void evaluateSHBasis(float x, float y, float z, int degree, inout float result[MAX_SH_COEFFS])
{
    result[0] = 0.28209479;

    if (degree < 1) return;

    result[1] = 0.48860251 * y;
    result[2] = 0.48860251 * z;
    result[3] = 0.48860251 * x;

    if (degree < 2) return;

    float xx = x * x, yy = y * y, zz = z * z;
    result[4] = 1.09254843 * x * y;
    result[5] = 1.09254843 * y * z;
    result[6] = 0.31539157 * (3.0 * zz - 1.0);
    result[7] = 1.09254843 * x * z;
    result[8] = 0.54627422 * (xx - yy);

    if (degree < 3) return;

    result[9]  = 0.59004359 * y * (3.0 * xx - yy);
    result[10] = 2.89061144 * x * y * z;
    result[11] = 0.45704579 * y * (5.0 * zz - 1.0);
    result[12] = 0.37317633 * z * (5.0 * zz - 3.0);
    result[13] = 0.45704579 * x * (5.0 * zz - 1.0);
    result[14] = 1.44530572 * z * (xx - yy);
    result[15] = 0.59004359 * x * (xx - 3.0 * yy);

    if (degree < 4) return;

    result[16] = 2.50334294 * x * y * (xx - yy);
    result[17] = 1.77013077 * y * z * (3.0 * xx - yy);
    result[18] = 0.94617469 * x * y * (7.0 * zz - 1.0);
    result[19] = 0.66904654 * y * z * (7.0 * zz - 3.0);
    result[20] = 0.10578554 * (35.0 * zz * zz - 30.0 * zz + 3.0);
    result[21] = 0.66904654 * x * z * (7.0 * zz - 3.0);
    result[22] = 0.47308735 * (xx - yy) * (7.0 * zz - 1.0);
    result[23] = 1.77013077 * x * z * (xx - 3.0 * yy);
    result[24] = 0.62583574 * (xx * xx - 6.0 * xx * yy + yy * yy);
}
