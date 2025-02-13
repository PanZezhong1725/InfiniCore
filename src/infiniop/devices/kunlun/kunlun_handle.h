#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__


#include "infinicore.h"
#include "infiniop/handle.h"

struct InfiniopKunlunHandle;
typedef struct InfiniopKunlunHandle *infiniopKunlunHandle_t;

infiniopStatus_t createKunlunHandle(infiniopKunlunHandle_t *handle_ptr, int device_id);
infiniopStatus_t deleteKunlunHandle(infiniopKunlunHandle_t handle);

#endif
