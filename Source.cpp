#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "DeviceTable.h"
#include "Table.h"
#include <cstdlib>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

int main() {
    std::ifstream t("program.cu");
    std::string gpu_program((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    std::cout << gpu_program << std::endl;

    // Create an instance of nvrtcProgram
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                                       gpu_program.c_str(),   // buffer
                                       "prog.cu",     // name
                                       0,             // numHeaders
                                       NULL,          // headers
                                       NULL));        // includeNames

    // add all name expressions for kernels
    std::vector<std::string> kernel_name_vec;
    std::vector<std::string> variable_name_vec;
    std::vector<int> variable_initial_value;

    std::vector<double> expected_result;

    // note the name expressions are parsed as constant expressions
    kernel_name_vec.push_back("getValueOneCol");
    expected_result.push_back(2.7);

    kernel_name_vec.push_back("getValueMultipleCols");
    expected_result.push_back(6.5);

    // add kernel name expressions to NVRTC. Note this must be done before
    // the program is compiled.
    for (size_t i = 0; i < kernel_name_vec.size(); ++i)
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_name_vec[i].c_str()));

    // add expressions for  __device__ / __constant__ variables to NVRTC

    for (size_t i = 0; i < variable_name_vec.size(); ++i)
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, variable_name_vec[i].c_str()));

    std::string cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path.empty()) {
        std::cerr << "CUDA path not found!\n";
        exit(1);
    }
    std::string path = "--include-path=" + cuda_path + "/include";

    std::vector<const char *> compileOptions;
    compileOptions.push_back("--include-path=.");
    compileOptions.push_back(path.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    compileOptions.size(),     // numOptions
                                                    compileOptions.data()); // options
// Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    std::cout << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }
    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    // Load the generated PTX
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

    CUdeviceptr dResult;
    double hResult = 0;
    CUDA_SAFE_CALL(cuMemAlloc(&dResult, sizeof(hResult)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dResult, &hResult, sizeof(hResult)));

    // for each of the __device__/__constant__ variable address
    // expressions provided to NVRTC, extract the lowered name for the
    // corresponding variable, and set its value
    for (size_t i = 0; i < variable_name_vec.size(); ++i) {
        const char *name;

        // note: this call must be made after NVRTC program has been
        // compiled and before it has been destroyed.
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                prog,
                variable_name_vec[i].c_str(), // name expression
                &name                         // lowered name
        ));
        double initial_value = variable_initial_value[i];

        // get pointer to variable using lowered name, and set its
        // initial value
        CUdeviceptr variable_addr;
        CUDA_SAFE_CALL(cuModuleGetGlobal(&variable_addr, NULL, module, name));
        CUDA_SAFE_CALL(cuMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value)));
    }

    Table *table1 = new Table(4);
    *table1 << 1 << 9 << 2 << 2 << 3 << 3 << 4 << 4;

    DeviceTable deviceTable1(4, table1->allocate());

    CUdeviceptr deviceTable1ptr;
    CUDA_SAFE_CALL(cuMemAlloc(&deviceTable1ptr, sizeof(deviceTable1)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(deviceTable1ptr, &deviceTable1, sizeof(deviceTable1)));

    Table *table2 = new Table(2, 2);
    *table2 << 3 << 4 << 1 << 5 << 6 << 2 << 7 << 8;

    DeviceTable deviceTable2(2,2, table2->allocate());
    CUdeviceptr deviceTable2ptr;
    CUDA_SAFE_CALL(cuMemAlloc(&deviceTable2ptr, sizeof(deviceTable2)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(deviceTable2ptr, &deviceTable2, sizeof(deviceTable2)));

    // for each of the kernel name expressions previously provided to NVRTC,
    // extract the lowered name for corresponding __global__ function,
    // and launch it.

    for (size_t i = 0; i < kernel_name_vec.size(); ++i) {
        const char *name;

        // note: this call must be made after NVRTC program has been
        // compiled and before it has been destroyed.
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                prog,
                kernel_name_vec[i].c_str(), // name expression
                &name                // lowered name
        ));

        // get pointer to kernel from loaded PTX
        CUfunction kernel;
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, name));

        // launch the kernel
        std::cout << "\nlaunching " << name << " ("
                  << kernel_name_vec[i] << ")" << std::endl;

        if (i == 0) {
            void *args[] = {&deviceTable1ptr, &dResult};
            CUDA_SAFE_CALL(
                    cuLaunchKernel(kernel,
                                   1, 1, 1,             // grid dim
                                   1, 1, 1,             // block dim
                                   0, NULL,             // shared mem and stream
                                   args, 0));           // arguments
        } else {
            void *args[] = {&deviceTable2ptr, &dResult};
            CUDA_SAFE_CALL(
                    cuLaunchKernel(kernel,
                                   1, 1, 1,             // grid dim
                                   1, 1, 1,             // block dim
                                   0, NULL,             // shared mem and stream
                                   args, 0));           // arguments
        }
        CUDA_SAFE_CALL(cuCtxSynchronize());

        // Retrieve the result
        CUDA_SAFE_CALL(cuMemcpyDtoH(&hResult, dResult, sizeof(hResult)));
        // check against expected value
        std::cout << "\n Expected result = " << expected_result[i]
                  << " , actual result = " << hResult << std::endl;

    }  // for

    delete table1;
    delete table2;
    CUDA_SAFE_CALL(cuMemFree(deviceTable1ptr));
    CUDA_SAFE_CALL(cuMemFree(deviceTable2ptr));
    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dResult));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    cudaDeviceReset();

    return 0;
}
