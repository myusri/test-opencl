#include <CL/cl.h>
#include <string.h>
#include <stdio.h>

static char* read_kernel_source(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel source file: %s\n", filename);
        exit(-1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *source = (char*)malloc(size + 1);
    if (fread(source, 1, size, file) != size) {
        fprintf(stderr, "Failed to read kernel source file: %s\n", filename);
        exit(-1);
    }
    source[size] = '\0';
    fclose(file);
    return source;
}

int main()
{
  // Define two 8x8 matrices for multiplication.
  float A[64] = {
    1, 2, 3, 4, 5, 6, 7, 8,
    8, 7, 6, 5, 4, 3, 2, 1,
    2, 3, 4, 5, 6, 7, 8, 9,
    9, 8, 7, 6, 5, 4, 3, 2,
    3, 4, 5, 6, 7, 8, 9, 10,
    10, 9, 8, 7, 6, 5, 4, 3,
    4, 5, 6, 7, 8, 9, 10, 11,
    11, 10, 9, 8, 7, 6, 7, 4,
  };
  float B[64] = {
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 1,
  };
  float C[64] = { 0 }; // all will be zero

  // Set up OpenCL environment.
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue_properties properties[] = { 0 };
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);
  
  // Load and compile the kernel.
  char* kernel_source = read_kernel_source("matrix-mul-kernel.cl");
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, NULL);
  free(kernel_source);

  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "matrix_mul", NULL);

  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, NULL);
  cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, NULL);
  cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 64, NULL, NULL);

  clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float) * 64, A, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float) * 64, B, 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

  size_t global_size[2] = {8, 8};
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * 64, C, 0, NULL, NULL);

  // Since B is a unit matrix, C should equal A.
  int status = 0;
  if (memcmp(A, C, sizeof(A)) != 0) {
    status = -1;
    printf("OpenCL 8x8 matrix multiplication failed\n");
  }
  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return status;
}
