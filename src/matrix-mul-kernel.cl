// 8x8 matrix multiplication kernel, C[row, col] = A[row] .* B[col]
__kernel void matrix_mul(
  const __global float* A, const __global float* B, __global float* C)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0;
    for(int k = 0; k < 8; k++)
    {
        sum += A[row * 8 + k] * B[k * 8 + col];
    }
    C[row * 8 + col] = sum;
}
