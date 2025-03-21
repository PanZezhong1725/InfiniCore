#include "test.h"
#include "../infinirt/ascend/infinirt_ascend.h"
#include "../infinirt/bang/infinirt_bang.h"
#include "../infinirt/cpu/infinirt_cpu.h"
#include "../infinirt/cuda/infinirt_cuda.cuh"
#include "../utils.h"
#include <cassert>
#include <cstring>
#include <iostream>

bool test_memcpy(infiniDevice_t device, size_t dataSize) {

    // 获取设备总数
    std::vector<int> count_array(INFINI_DEVICE_TYPE_COUNT, 0);
    if (infinirtGetAllDeviceCount(count_array.data()) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get total device count." << std::endl;
        return false;
    }

    std::cout << "Available devices for device type " << device << ": "
              << count_array[device] << std::endl;
    int total_devices = count_array[device];

    for (int device_id = 0; device_id < total_devices; ++device_id) {
        infiniDevice_t device = INFINI_DEVICE_NVIDIA; // 使用parse

        std::cout << "\n==============================================" << std::endl;
        std::cout << "Testing memcpy on Device ID: " << device_id << std::endl;
        std::cout << "==============================================" << std::endl;

        // 设置当前设备
        std::cout << "[Device " << device_id << "] Setting device..." << std::endl;
        if (infinirtSetDevice(device, device_id) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to set device." << std::endl;
            continue;
        }

        // 分配主机内存
        std::cout << "[Device " << device_id << "] Allocating host memory: " << dataSize * sizeof(float) << " bytes" << std::endl;
        std::vector<float> hostData(dataSize, 1.23f);
        std::vector<float> hostCopy(dataSize, 0.0f);

        // 分配设备内存
        void *deviceSrc = nullptr, *deviceDst = nullptr;
        size_t dataSizeInBytes = dataSize * sizeof(float);

        std::cout << "[Device " << device_id << "] Allocating device memory: " << dataSizeInBytes << " bytes" << std::endl;
        if (infinirtMalloc(&deviceSrc, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to allocate device memory for deviceSrc." << std::endl;
            continue;
        }

        if (infinirtMalloc(&deviceDst, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to allocate device memory for deviceDst." << std::endl;
            infinirtFree(deviceSrc);
            continue;
        }

        // 复制数据到设备
        std::cout << "[Device " << device_id << "] Copying data from host to device..." << std::endl;
        if (infinirtMemcpy(deviceSrc, hostData.data(), dataSizeInBytes, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to copy data from host to device." << std::endl;
            infinirtFree(deviceSrc);
            infinirtFree(deviceDst);
            continue;
        }

        // 设备内存间复制
        std::cout << "[Device " << device_id << "] Copying data between device memory (D2D)..." << std::endl;
        if (infinirtMemcpy(deviceDst, deviceSrc, dataSizeInBytes, INFINIRT_MEMCPY_D2D) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to copy data from device to device." << std::endl;
            infinirtFree(deviceSrc);
            infinirtFree(deviceDst);
            continue;
        }

        // 设备数据复制回主机
        std::cout << "[Device " << device_id << "] Copying data from device back to host..." << std::endl;
        if (infinirtMemcpy(hostCopy.data(), deviceDst, dataSizeInBytes, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
            std::cerr << "[Device " << device_id << "] Failed to copy data from device to host." << std::endl;
            infinirtFree(deviceSrc);
            infinirtFree(deviceDst);
            continue;
        }

        // 数据验证
        std::cout << "[Device " << device_id << "] Validating copied data..." << std::endl;
        if (std::memcmp(hostData.data(), hostCopy.data(), dataSizeInBytes) != 0) {
            std::cerr << "[Device " << device_id << "] Data mismatch between hostData and hostCopy." << std::endl;
            infinirtFree(deviceSrc);
            infinirtFree(deviceDst);
            continue;
        }

        std::cout << "[Device " << device_id << "] Data copied correctly!" << std::endl;

        // 释放设备内存
        std::cout << "[Device " << device_id << "] Freeing device memory..." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);

        std::cout << "[Device " << device_id << "] Memory copy test PASSED!" << std::endl;
    }

    return true;
}

bool test_setDevice(infiniDevice_t device, int device_id) {

    struct TestCase {
        infiniDevice_t device;
        int device_id;
        infiniStatus_t expected_status;
    };

    std::vector<TestCase> testCases = {
        {INFINI_DEVICE_CPU, device_id, INFINI_STATUS_SUCCESS},
        {device, device_id, INFINI_STATUS_SUCCESS}};

    for (const auto &test : testCases) {
        std::cout << "Setting device: " << test.device << " with ID: " << test.device_id << std::endl;

        infiniStatus_t status = infinirtSetDevice(test.device, test.device_id);
        std::cout << "Status after setting device: " << status << std::endl;

        if (status != test.expected_status) {
            std::cerr << "Expected status: " << test.expected_status << ", but got: " << status << std::endl;
            return false;
        }
    }
    return true;
}
