#include "test.h"

struct ParsedArgs {
    infiniDevice_t device_type = INFINI_DEVICE_CPU; // 默认 CPU
};

void printUsage() {
    std::cout << "Usage:\n"
              << "  infinirt-test [--<device>]\n\n"
              << "  --<device>   Specify device type.\n"
              << "               Available devices: cpu, nvidia, cambricon, ascend,\n"
              << "               metax, moore, iluvatar, kunlun, sugon\n"
              << "               Default is CPU.\n"
              << std::endl;
    exit(EXIT_FAILURE);
}

#define PARSE_DEVICE(FLAG, DEVICE) \
    else if (arg == FLAG) {        \
        args.device_type = DEVICE; \
    }

ParsedArgs parseArgs(int argc, char *argv[]) {
    ParsedArgs args;

    if (argc < 2) {
        return args; // 默认使用 CPU
    }

    std::string arg = argv[1];
    if (arg == "--help" || arg == "-h") {
        printUsage();
    }

    try {
        if (arg == "--cpu") {
            args.device_type = INFINI_DEVICE_CPU;
        }
        PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
        PARSE_DEVICE("--cambricon", INFINI_DEVICE_CAMBRICON)
        PARSE_DEVICE("--ascend", INFINI_DEVICE_ASCEND)
        PARSE_DEVICE("--metax", INFINI_DEVICE_METAX)
        PARSE_DEVICE("--moore", INFINI_DEVICE_MOORE)
        PARSE_DEVICE("--iluvatar", INFINI_DEVICE_ILUVATAR)
        PARSE_DEVICE("--kunlun", INFINI_DEVICE_KUNLUN)
        PARSE_DEVICE("--sugon", INFINI_DEVICE_SUGON)
        else {
            printUsage();
        }
    } catch (const std::exception &) {
        printUsage();
    }

    return args;
}
int main(int argc, char *argv[]) {

    ParsedArgs args = parseArgs(argc, argv);
    std::cout << "\nTesting Device: " << args.device_type << std::endl;
    infiniDevice_t device = args.device_type;

    // test_setDevice
    std::cout << "===== Start test_setDevice =====" << std::endl;
    if (!test_setDevice(device, 0)) {
        std::cerr << "test_setDevice Test failed." << std::endl;
        return 1;
    }
    std::cout << "===== test_setDevice Completed =====\n"
              << std::endl;

    // test_memcpy
    std::cout << "===== Start test_memcpy =====" << std::endl;
    size_t dataSize = 1024;
    if (!test_memcpy(device, dataSize)) {
        std::cerr << "test_memcpy Test failed." << std::endl;
        return 1;
    }
    std::cout << "===== test_memcpy Completed =====\n"
              << std::endl;

    return 0;
}
