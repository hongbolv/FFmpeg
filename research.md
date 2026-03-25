# FFmpeg DNN Module 深度研究报告

## libavfilter → DNN Interface → OpenVINO Backend → OpenVINO C Library 调用链分析与图像超分辨率（SR）模型集成指南

---

## 目录

1. [FFmpeg 项目概述与架构](#1-ffmpeg-项目概述与架构)
2. [DNN 模块整体架构](#2-dnn-模块整体架构)
3. [libavfilter 视频滤镜层](#3-libavfilter-视频滤镜层)
4. [DNN Interface 接口层](#4-dnn-interface-接口层)
5. [OpenVINO Backend 后端层](#5-openvino-backend-后端层)
6. [OpenVINO C Library 调用](#6-openvino-c-library-调用)
7. [完整调用链详解](#7-完整调用链详解)
8. [数据流与 I/O 处理](#8-数据流与-io-处理)
9. [异步执行机制](#9-异步执行机制)
10. [大尺寸图片分块 SR 处理分析](#10-大尺寸图片分块-sr-处理分析)
11. [新图像 SR 模型集成步骤](#11-新图像-sr-模型集成步骤)
12. [附录：关键数据结构参考](#12-附录关键数据结构参考)

---

## 1. FFmpeg 项目概述与架构

### 1.1 项目结构

FFmpeg 是一个跨平台的多媒体处理框架，核心库包括：

```
FFmpeg/
├── libavcodec/       # 编解码库
├── libavformat/      # 封装/解封装库
├── libavfilter/      # 音视频滤镜库（DNN模块所在位置）
│   ├── dnn/          # DNN 后端实现
│   ├── dnn_interface.h         # DNN 接口头文件
│   ├── dnn_filter_common.h/c   # DNN 滤镜公共层
│   ├── vf_dnn_processing.c     # DNN 处理滤镜（SR/去噪等）
│   ├── vf_dnn_detect.c         # DNN 目标检测滤镜
│   └── vf_dnn_classify.c       # DNN 分类滤镜
├── libavutil/        # 工具库
├── libswscale/       # 图像缩放库
├── libswresample/    # 音频重采样库
├── libavdevice/      # 设备输入输出库
├── fftools/          # 命令行工具（ffmpeg, ffprobe等）
└── doc/              # 文档
```

### 1.2 DNN 相关文件一览

| 文件路径 | 作用 |
|---------|------|
| `libavfilter/dnn_interface.h` | DNN 接口定义（核心头文件） |
| `libavfilter/dnn/dnn_interface.c` | DNN 模块初始化与后端路由 |
| `libavfilter/dnn_filter_common.h` | DNN 滤镜公共宏和函数声明 |
| `libavfilter/dnn_filter_common.c` | DNN 滤镜公共初始化/执行/销毁逻辑 |
| `libavfilter/dnn/dnn_backend_openvino.c` | OpenVINO 后端实现（1621行） |
| `libavfilter/dnn/dnn_backend_tf.c` | TensorFlow 后端实现 |
| `libavfilter/dnn/dnn_backend_torch.cpp` | PyTorch(LibTorch) 后端实现 |
| `libavfilter/dnn/dnn_backend_common.h` | 后端公共结构与函数 |
| `libavfilter/dnn/dnn_backend_common.c` | 后端公共实现（任务填充/异步执行） |
| `libavfilter/dnn/dnn_io_proc.h` | I/O 处理接口 |
| `libavfilter/dnn/dnn_io_proc.c` | AVFrame ↔ DNNData 转换实现 |
| `libavfilter/dnn/queue.h/c` | 双端队列实现 |
| `libavfilter/dnn/safe_queue.h/c` | 线程安全队列实现 |
| `libavfilter/vf_dnn_processing.c` | 通用 DNN 图像处理滤镜 |
| `libavfilter/vf_dnn_detect.c` | DNN 目标检测滤镜 |
| `libavfilter/vf_dnn_classify.c` | DNN 分类滤镜 |

---

## 2. DNN 模块整体架构

### 2.1 分层架构

FFmpeg 的 DNN 模块采用分层设计，自上而下分为四层：

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Video Filters (vf_dnn_processing / detect / classify) │
│  视频滤镜层：面向用户的滤镜接口                                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: DNN Filter Common (dnn_filter_common)                  │
│  滤镜公共层：统一封装 DNN 初始化/执行/结果获取                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: DNN Interface (dnn_interface)                          │
│  DNN 接口层：定义 DNNModule/DNNModel 抽象接口                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: DNN Backend (openvino / tensorflow / torch)            │
│  后端层：具体推理框架实现                                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: External Library (OpenVINO C API / TF C API / LibTorch)│
│  外部库层：第三方推理引擎                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 后端类型

```c
typedef enum {
    DNN_TF = 1,        // TensorFlow 后端
    DNN_OV = 1 << 1,   // OpenVINO 后端
    DNN_TH = 1 << 2    // LibTorch (PyTorch) 后端
} DNNBackendType;
```

### 2.3 功能类型

```c
typedef enum {
    DFT_NONE,
    DFT_PROCESS_FRAME,      // 全帧处理（超分/去噪/风格迁移）
    DFT_ANALYTICS_DETECT,   // 目标检测
    DFT_ANALYTICS_CLASSIFY, // 分类
} DNNFunctionType;
```

---

## 3. libavfilter 视频滤镜层

### 3.1 vf_dnn_processing.c — 通用 DNN 处理滤镜

这是图像超分辨率（SR）模型最常使用的滤镜入口。

#### 3.1.1 上下文结构

```c
typedef struct DnnProcessingContext {
    const AVClass *class;
    DnnContext dnnctx;              // DNN 上下文（包含模型、后端等信息）
    struct SwsContext *sws_uv_scale; // YUV格式UV通道缩放上下文
    int sws_uv_height;
} DnnProcessingContext;
```

#### 3.1.2 滤镜选项

```c
static const AVOption dnn_processing_options[] = {
    { "dnn_backend", "DNN backend", ..., { .i64 = DNN_TF }, ... },
    { "tensorflow",  "tensorflow backend flag", ..., { .i64 = DNN_TF }, ... },
    { "openvino",    "openvino backend flag",   ..., { .i64 = DNN_OV }, ... },
    { "torch",       "torch backend flag",      ..., { .i64 = DNN_TH }, ... },
    { NULL }
};
```

#### 3.1.3 初始化流程

```c
static av_cold int init(AVFilterContext *context) {
    DnnProcessingContext *ctx = context->priv;
    // 以 DFT_PROCESS_FRAME 类型初始化 DNN
    return ff_dnn_init(&ctx->dnnctx, DFT_PROCESS_FRAME, context);
}
```

#### 3.1.4 核心处理流程（activate 函数）

```
activate() {
    1. 从输入链路消费帧: ff_inlink_consume_frame()
    2. 分配输出视频缓冲: ff_get_video_buffer()
    3. 复制帧属性: av_frame_copy_props()
    4. 执行 DNN 模型: ff_dnn_execute_model()
    5. 获取异步结果: ff_dnn_get_result()
    6. 处理 UV 通道（YUV格式）: copy_uv_planes()
    7. 输出帧: ff_filter_frame()
}
```

#### 3.1.5 输入/输出配置

- **config_input**: 获取模型输入信息，校验输入尺寸和通道数
- **config_output**: 通过 `ff_dnn_get_output()` 获取模型输出尺寸（支持SR放大）

#### 3.1.6 支持的像素格式

```c
static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};
```

### 3.2 vf_dnn_detect.c — 目标检测滤镜

- 使用 `DFT_ANALYTICS_DETECT` 功能类型
- 设置检测后处理回调：`ff_dnn_set_detect_post_proc(&ctx->dnnctx, dnn_detect_post_proc)`
- 支持 SSD、YOLOv1/v2/v3/v4 模型类型
- 后处理包含 NMS（非极大值抑制）

### 3.3 vf_dnn_classify.c — 分类滤镜

- 使用 `DFT_ANALYTICS_CLASSIFY` 功能类型
- 设置分类后处理回调：`ff_dnn_set_classify_post_proc(&ctx->dnnctx, dnn_classify_post_proc)`
- 依赖检测滤镜提供的 bounding box 数据

---

## 4. DNN Interface 接口层

### 4.1 核心接口：DNNModule

`DNNModule` 是后端抽象接口，每个后端（OpenVINO/TF/Torch）实现一套：

```c
struct DNNModule {
    const AVClass clazz;        // AVClass 用于选项系统
    DNNBackendType type;        // 后端类型标识

    // 加载模型（从文件加载，返回 DNNModel 指针）
    DNNModel *(*load_model)(DnnContext *ctx, DNNFunctionType func_type,
                            AVFilterContext *filter_ctx);

    // 执行模型推理
    int (*execute_model)(const DNNModel *model, DNNExecBaseParams *exec_params);

    // 获取异步推理结果
    DNNAsyncStatusType (*get_result)(const DNNModel *model,
                                     AVFrame **in, AVFrame **out);

    // 刷新所有待处理任务
    int (*flush)(const DNNModel *model);

    // 释放模型资源
    void (*free_model)(DNNModel **model);
};
```

### 4.2 核心数据模型：DNNModel

```c
typedef struct DNNModel {
    AVFilterContext *filter_ctx;   // 关联的滤镜上下文
    DNNFunctionType func_type;    // 功能类型

    // 获取模型输入信息
    int (*get_input)(struct DNNModel *model, DNNData *input, const char *input_name);

    // 获取模型输出尺寸
    int (*get_output)(struct DNNModel *model, const char *input_name,
                      int input_width, int input_height,
                      const char *output_name, int *output_width, int *output_height);

    // 预处理回调：AVFrame → DNNData
    FramePrePostProc frame_pre_proc;

    // 后处理回调：DNNData → AVFrame
    FramePrePostProc frame_post_proc;

    // 检测后处理回调
    DetectPostProc detect_post_proc;

    // 分类后处理回调
    ClassifyPostProc classify_post_proc;
} DNNModel;
```

### 4.3 DNN 上下文：DnnContext

```c
typedef struct DnnContext {
    const AVClass *clazz;
    DNNModel *model;                // 当前加载的模型
    char *model_filename;           // 模型文件路径
    DNNBackendType backend_type;    // 后端类型
    char *model_inputname;          // 模型输入节点名
    char *model_outputnames_string; // 模型输出节点名（字符串）
    char **model_outputnames;       // 解析后的输出节点名数组
    uint32_t nb_outputs;            // 输出数量
    const DNNModule *dnn_module;    // 当前使用的后端模块

    int nireq;      // 并发推理请求数
    int async;       // 是否启用异步推理
    char *device;    // 推理设备（CPU/GPU等）

    // 后端专有选项
    OVOptions ov_option;    // OpenVINO 选项
    TFOptions tf_option;    // TensorFlow 选项
    THOptions torch_option; // LibTorch 选项
} DnnContext;
```

### 4.4 后端路由：ff_get_dnn_module()

```c
// 文件：libavfilter/dnn/dnn_interface.c
const DNNModule *ff_get_dnn_module(DNNBackendType backend_type, void *log_ctx) {
    // 遍历 dnn_backend_info_list，按 type 匹配后端
    for (int i = 1; i < FF_ARRAY_ELEMS(dnn_backend_info_list); i++) {
        if (dnn_backend_info_list[i].module->type == backend_type)
            return dnn_backend_info_list[i].module;
    }
    return NULL;
}
```

后端注册列表（编译期条件编译）：

```c
static const DnnBackendInfo dnn_backend_info_list[] = {
    {0, .class = &dnn_base_class},
#if CONFIG_LIBTENSORFLOW
    {offsetof(DnnContext, tf_option), .module = &ff_dnn_backend_tf},
#endif
#if CONFIG_LIBOPENVINO
    {offsetof(DnnContext, ov_option), .module = &ff_dnn_backend_openvino},
#endif
#if CONFIG_LIBTORCH
    {offsetof(DnnContext, torch_option), .module = &ff_dnn_backend_torch},
#endif
};
```

### 4.5 dnn_filter_common.c — 统一封装层

此文件将 `DNNModule` 的函数指针调用封装为直观的函数名：

```c
// 初始化 DNN（解析选项、获取后端模块、加载模型）
int ff_dnn_init(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx) {
    ctx->dnn_module = ff_get_dnn_module(ctx->backend_type, filter_ctx);
    ctx->model = ctx->dnn_module->load_model(ctx, func_type, filter_ctx);
    return 0;
}

// 执行模型推理
int ff_dnn_execute_model(DnnContext *ctx, AVFrame *in_frame, AVFrame *out_frame) {
    DNNExecBaseParams exec_params = { .input_name = ..., .in_frame = in_frame, ... };
    return ctx->dnn_module->execute_model(ctx->model, &exec_params);
}

// 获取推理结果
DNNAsyncStatusType ff_dnn_get_result(DnnContext *ctx, AVFrame **in_frame, AVFrame **out_frame) {
    return ctx->dnn_module->get_result(ctx->model, in_frame, out_frame);
}

// 获取模型输入信息
int ff_dnn_get_input(DnnContext *ctx, DNNData *input) {
    return ctx->model->get_input(ctx->model, input, ctx->model_inputname);
}

// 获取模型输出尺寸（用于 SR 等会改变分辨率的模型）
int ff_dnn_get_output(DnnContext *ctx, int input_width, int input_height,
                      int *output_width, int *output_height) {
    return ctx->model->get_output(ctx->model, ctx->model_inputname,
                                  input_width, input_height,
                                  output_name, output_width, output_height);
}

// 释放模型
void ff_dnn_uninit(DnnContext *ctx) {
    ctx->dnn_module->free_model(&ctx->model);
}
```

---

## 5. OpenVINO Backend 后端层

### 5.1 文件位置

`libavfilter/dnn/dnn_backend_openvino.c`（1621行）

### 5.2 核心数据结构

#### 5.2.1 OVModel — OpenVINO 模型封装

```c
typedef struct OVModel {
    DNNModel model;           // 继承 DNNModel（嵌入作为第一个成员）
    DnnContext *ctx;          // DNN 上下文引用

    // OpenVINO 2.0 API (HAVE_OPENVINO2)
    ov_core_t *core;                        // OpenVINO 运行时核心
    ov_model_t *ov_model;                   // 模型对象
    ov_compiled_model_t *compiled_model;    // 编译后的模型
    ov_output_const_port_t *input_port;     // 输入端口
    ov_output_const_port_t **output_ports;  // 输出端口数组
    ov_preprocess_prepostprocessor_t *preprocess; // 预处理器

    SafeQueue *request_queue;  // 推理请求队列（线程安全）
    Queue *task_queue;         // 任务队列
    Queue *lltask_queue;       // 底层任务队列（一个task可分多个lltask）
    int nb_outputs;            // 输出端口数量
} OVModel;
```

#### 5.2.2 OVRequestItem — 推理请求项

```c
typedef struct OVRequestItem {
    LastLevelTaskItem **lltasks;     // 关联的底层任务列表
    uint32_t lltask_count;          // 底层任务数量

    // OpenVINO 2.0 API
    ov_infer_request_t *infer_request;  // OpenVINO 推理请求对象
    ov_callback_t callback;              // 推理完成回调
} OVRequestItem;
```

### 5.3 OpenVINO 后端选项

```c
typedef struct OVOptions {
    const AVClass *clazz;
    int batch_size;         // 批大小（默认1）
    int input_resizable;    // 输入是否可调整大小（默认0）
    DNNLayout layout;       // 输入布局（NCHW/NHWC/NONE）
    float scale;            // 输入缩放因子（用于预处理，默认255或1）
    float mean;             // 输入均值偏移（用于预处理）
} OVOptions;
```

### 5.4 后端模块注册

```c
const DNNModule ff_dnn_backend_openvino = {
    .clazz          = DNN_DEFINE_CLASS(dnn_openvino),
    .type           = DNN_OV,
    .load_model     = dnn_load_model_ov,     // 加载模型
    .execute_model  = dnn_execute_model_ov,  // 执行推理
    .get_result     = dnn_get_result_ov,     // 获取结果
    .flush          = dnn_flush_ov,          // 刷新待处理
    .free_model     = dnn_free_model_ov,     // 释放资源
};
```

### 5.5 关键函数详解

#### 5.5.1 dnn_load_model_ov() — 模型加载

```
dnn_load_model_ov(ctx, func_type, filter_ctx)
│
├── 1. 分配 OVModel 结构体
│
├── 2. ov_core_create(&core)              // 创建 OpenVINO 核心
│
├── 3. ov_core_read_model(core,           // 从文件读取模型
│      model_filename, NULL, &ovmodel)
│
├── 4. 设置函数指针:
│      model->get_input  = &get_input_ov
│      model->get_output = &get_output_ov
│      model->filter_ctx = filter_ctx
│      model->func_type  = func_type
│
└── 5. 返回 DNNModel 指针
       ⚠ 注意：此时模型尚未编译，编译延迟到首次执行时
```

#### 5.5.2 init_model_ov() — 模型初始化（延迟编译）

这是OpenVINO后端最核心的初始化函数，在首次调用 `execute_model` 或 `get_output` 时触发：

```
init_model_ov(ov_model, input_name, output_names, nb_outputs)
│
├── 1. 设置 scale/mean 参数
│      默认 scale=255（DFT_PROCESS_FRAME），scale=1（其他）
│
├── 2. 创建预处理器
│      ov_preprocess_prepostprocessor_create()
│
├── 3. 配置输入预处理
│      ├── 获取输入信息: ov_preprocess_prepostprocessor_get_input_info()
│      ├── 获取张量信息: ov_preprocess_input_info_get_tensor_info()
│      ├── 设置输入布局为NHWC: ov_preprocess_input_tensor_info_set_layout(NHWC)
│      ├── 设置模型布局: ov_preprocess_input_model_info_set_layout(NCHW/NHWC)
│      ├── 设置输入数据类型为U8: ov_preprocess_input_tensor_info_set_element_type(U8)
│      └── 添加预处理步骤（如 scale/mean）:
│          ├── ov_preprocess_preprocess_steps_convert_element_type(F32)
│          ├── ov_preprocess_preprocess_steps_mean(mean)
│          └── ov_preprocess_preprocess_steps_scale(scale)
│
├── 4. 配置输出后处理
│      ├── 获取输出信息: ov_preprocess_prepostprocessor_get_output_info()
│      └── 设置输出数据类型: ov_preprocess_output_set_element_type(F32/U8)
│
├── 5. 构建预处理模型
│      ov_preprocess_prepostprocessor_build() → 更新 ov_model->ov_model
│
├── 6. 获取输出端口
│      ov_model_const_output_by_name() 或 ov_model_const_output_by_index()
│
├── 7. 编译模型到目标设备
│      ov_core_compile_model(core, ov_model, device, 0, &compiled_model)
│
├── 8. 创建推理请求队列
│      ├── 确定并发请求数: nireq = cpu_count/2 + 1
│      ├── 创建 SafeQueue: ff_safe_queue_create()
│      └── 循环创建 OVRequestItem:
│          ├── 设置回调函数: callback.callback_func = infer_completion_callback
│          ├── 创建推理请求: ov_compiled_model_create_infer_request()
│          └── 分配 lltasks 数组
│
├── 9. 创建任务队列
│      ├── task_queue = ff_queue_create()
│      └── lltask_queue = ff_queue_create()
│
└── 10. 返回 0（成功）
```

#### 5.5.3 dnn_execute_model_ov() — 执行推理

```
dnn_execute_model_ov(model, exec_params)
│
├── 1. 参数检查: ff_check_exec_params()
│
├── 2. 延迟初始化（首次调用时）:
│      if (!compiled_model) → init_model_ov()
│
├── 3. 创建任务: av_malloc(sizeof(TaskItem))
│      ff_dnn_fill_task(task, exec_params, ov_model, async, do_ioproc=1)
│
├── 4. 入队: ff_queue_push_back(task_queue, task)
│
├── 5. 提取底层任务: extract_lltask_from_task()
│      ├── DFT_PROCESS_FRAME: 一个 task → 一个 lltask
│      ├── DFT_ANALYTICS_DETECT: 一个 task → 一个 lltask
│      └── DFT_ANALYTICS_CLASSIFY: 一个 task → 多个 lltask（每个bbox一个）
│
├── 6a. 异步模式 (async=1):
│      while (lltask_queue >= batch_size) {
│          request = ff_safe_queue_pop_front(request_queue)
│          execute_model_ov(request, lltask_queue)
│      }
│
└── 6b. 同步模式 (async=0):
       request = ff_safe_queue_pop_front(request_queue)
       execute_model_ov(request, lltask_queue)
```

#### 5.5.4 execute_model_ov() — 实际推理执行

```
execute_model_ov(request, inferenceq)
│
├── 1. 填充模型输入: fill_model_input_ov()
│      ├── 获取输入端口形状: ov_const_port_get_shape()
│      ├── 获取数据类型: ov_port_get_element_type()
│      ├── 创建输入张量: ov_tensor_create()
│      ├── 获取张量数据指针: ov_tensor_data()
│      ├── 设置输入张量: ov_infer_request_set_input_tensor()
│      └── 数据预处理（按功能类型）:
│          ├── DFT_PROCESS_FRAME: ff_proc_from_frame_to_dnn() 或 frame_pre_proc()
│          ├── DFT_ANALYTICS_DETECT: ff_frame_to_dnn_detect()
│          └── DFT_ANALYTICS_CLASSIFY: ff_frame_to_dnn_classify()
│
├── 2a. 异步推理:
│      ├── ov_infer_request_set_callback(request, &callback)
│      └── ov_infer_request_start_async(request)
│
└── 2b. 同步推理:
       ├── ov_infer_request_infer(request)
       └── infer_completion_callback(request)  // 直接调用回调
```

#### 5.5.5 infer_completion_callback() — 推理完成回调

```
infer_completion_callback(args)
│
├── 1. 获取输出数据:
│      ├── ov_infer_request_get_tensor_by_const_port() → output_tensor
│      ├── ov_tensor_data(output_tensor) → outputs[i].data
│      ├── ov_tensor_get_shape(output_tensor) → output_shape
│      └── ov_port_get_element_type() → precision
│
├── 2. 填充输出 DNNData:
│      outputs[i].dt     = precision_to_datatype(precision)
│      outputs[i].layout = ctx->ov_option.layout
│      outputs[i].scale  = ctx->ov_option.scale
│      outputs[i].mean   = ctx->ov_option.mean
│      outputs[i].dims   = [batch, channels/height, height/width, width/channels]
│
├── 3. 后处理（按功能类型）:
│      ├── DFT_PROCESS_FRAME:
│      │   ├── frame_post_proc() — 自定义后处理
│      │   └── ff_proc_from_dnn_to_frame() — 默认后处理
│      ├── DFT_ANALYTICS_DETECT:
│      │   └── detect_post_proc() — 检测后处理
│      └── DFT_ANALYTICS_CLASSIFY:
│          └── classify_post_proc() — 分类后处理
│
├── 4. 标记推理完成: task->inference_done++
│
└── 5. 回收请求到队列: ff_safe_queue_push_back(requestq, request)
```

#### 5.5.6 get_input_ov() — 获取模型输入信息

```
get_input_ov(model, input, input_name)
│
├── 1. ov_model_const_input_by_name() 或 ov_model_const_input()
├── 2. ov_port_get_element_type() → precision
├── 3. ov_const_port_get_shape() → input_shape
├── 4. 填充 input->dims[0..3]
├── 5. 自动判断布局:
│      if dims[1] <= 3 → DL_NCHW
│      else → DL_NHWC
├── 6. input->dt = precision_to_datatype(precision)
└── 7. 如果 input_resizable，将宽高设为 -1
```

#### 5.5.7 get_output_ov() — 获取模型输出尺寸

这对SR模型至关重要，因为输出尺寸与输入不同：

```
get_output_ov(model, input_name, input_width, input_height,
              output_name, output_width, output_height)
│
├── 1. 如果 input_resizable:
│      ├── 创建 partial_shape 描述输入尺寸
│      └── ov_model_reshape_single_input() — reshape 模型
│
├── 2. 如果未编译: init_model_ov()
│
├── 3. 创建测试任务: ff_dnn_fill_gettingoutput_task()
│      （创建空的 in_frame/out_frame，不做实际I/O处理）
│
├── 4. 提取底层任务: extract_lltask_from_task()
│
├── 5. 执行推理: execute_model_ov()
│
├── 6. 从结果获取输出尺寸:
│      *output_width  = task.out_frame->width
│      *output_height = task.out_frame->height
│
└── 7. 释放临时帧
```

---

## 6. OpenVINO C Library 调用

### 6.1 API 版本

FFmpeg 同时支持两个版本的 OpenVINO C API：

```c
#if HAVE_OPENVINO2
#include <openvino/c/openvino.h>    // OpenVINO 2.0 C API
#else
#include <c_api/ie_c_api.h>         // OpenVINO 1.0 (Inference Engine) C API
#endif
```

### 6.2 OpenVINO 2.0 C API 调用一览

以下是 FFmpeg OpenVINO 后端使用的所有 OpenVINO 2.0 C API 函数：

#### 核心管理

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_core_create()` | 创建 OpenVINO 运行时核心 | `dnn_load_model_ov()` |
| `ov_core_read_model()` | 从文件读取模型 | `dnn_load_model_ov()` |
| `ov_core_compile_model()` | 编译模型到目标设备 | `init_model_ov()` |
| `ov_core_free()` | 释放核心对象 | `dnn_free_model_ov()` |

#### 模型操作

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_model_const_input_by_name()` | 按名称获取输入端口 | `fill_model_input_ov()`, `get_input_ov()` |
| `ov_model_const_input()` | 获取默认输入端口 | `fill_model_input_ov()`, `get_input_ov()` |
| `ov_model_const_output_by_name()` | 按名称获取输出端口 | `init_model_ov()` |
| `ov_model_const_output_by_index()` | 按索引获取输出端口 | `init_model_ov()` |
| `ov_model_outputs_size()` | 获取输出端口数量 | `init_model_ov()` |
| `ov_model_reshape_single_input()` | 重塑模型输入形状 | `get_output_ov()` |
| `ov_model_free()` | 释放模型对象 | `dnn_free_model_ov()` |

#### 预处理

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_preprocess_prepostprocessor_create()` | 创建预处理器 | `init_model_ov()` |
| `ov_preprocess_prepostprocessor_get_input_info()` | 获取输入预处理信息 | `init_model_ov()` |
| `ov_preprocess_input_info_get_tensor_info()` | 获取输入张量信息 | `init_model_ov()` |
| `ov_preprocess_input_tensor_info_set_layout()` | 设置输入张量布局 | `init_model_ov()` |
| `ov_preprocess_input_tensor_info_set_element_type()` | 设置输入数据类型 | `init_model_ov()` |
| `ov_preprocess_input_info_get_model_info()` | 获取模型输入信息 | `init_model_ov()` |
| `ov_preprocess_input_model_info_set_layout()` | 设置模型输入布局 | `init_model_ov()` |
| `ov_preprocess_input_info_get_preprocess_steps()` | 获取预处理步骤 | `init_model_ov()` |
| `ov_preprocess_preprocess_steps_convert_element_type()` | 添加类型转换步骤 | `init_model_ov()` |
| `ov_preprocess_preprocess_steps_mean()` | 添加均值减除步骤 | `init_model_ov()` |
| `ov_preprocess_preprocess_steps_scale()` | 添加缩放步骤 | `init_model_ov()` |
| `ov_preprocess_output_set_element_type()` | 设置输出数据类型 | `init_model_ov()` |
| `ov_preprocess_prepostprocessor_build()` | 构建预处理后的模型 | `init_model_ov()` |

#### 推理执行

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_compiled_model_create_infer_request()` | 创建推理请求 | `init_model_ov()` |
| `ov_infer_request_set_input_tensor()` | 设置输入张量 | `fill_model_input_ov()` |
| `ov_infer_request_set_callback()` | 设置完成回调（异步） | `execute_model_ov()` |
| `ov_infer_request_start_async()` | 启动异步推理 | `execute_model_ov()` |
| `ov_infer_request_infer()` | 同步推理 | `execute_model_ov()`, `dnn_flush_ov()` |
| `ov_infer_request_get_tensor_by_const_port()` | 获取输出张量 | `infer_completion_callback()` |

#### 张量操作

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_tensor_create()` | 创建输入张量 | `fill_model_input_ov()` |
| `ov_tensor_data()` | 获取张量数据指针 | `fill_model_input_ov()`, `infer_completion_callback()` |
| `ov_tensor_get_shape()` | 获取张量形状 | `infer_completion_callback()` |
| `ov_tensor_free()` | 释放张量 | 多处 |

#### 端口/形状操作

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_port_get_element_type()` | 获取端口数据类型 | 多处 |
| `ov_port_get_any_name()` | 获取端口名称 | `fill_model_input_ov()`, `init_model_ov()` |
| `ov_const_port_get_shape()` | 获取端口形状 | 多处 |
| `ov_shape_free()` | 释放形状对象 | 多处 |
| `ov_layout_create()` | 创建布局描述 | `init_model_ov()` |
| `ov_layout_free()` | 释放布局对象 | `init_model_ov()` |
| `ov_partial_shape_create()` | 创建部分形状 | `get_output_ov()` |
| `ov_shape_to_partial_shape()` | 形状转部分形状 | `get_output_ov()` |

#### 错误处理

| API 函数 | 用途 | 调用位置 |
|----------|------|---------|
| `ov_get_openvino_version()` | 获取版本信息 | `dnn_load_model_ov()` |
| `ov_version_free()` | 释放版本信息 | `dnn_load_model_ov()` |

### 6.3 错误码映射

```c
static const struct {
    ov_status_e status;
    int         av_err;
    const char *desc;
} ov2_errors[] = {
    { OK,                     0,                  "success"                },
    { GENERAL_ERROR,          AVERROR_EXTERNAL,   "general error"          },
    { NOT_IMPLEMENTED,        AVERROR(ENOSYS),    "not implemented"        },
    { PARAMETER_MISMATCH,     AVERROR(EINVAL),    "parameter mismatch"     },
    { REQUEST_BUSY,           AVERROR(EBUSY),     "request busy"           },
    { RESULT_NOT_READY,       AVERROR(EBUSY),     "result not ready"       },
    // ... 等等
};
```

---

## 7. 完整调用链详解

### 7.1 SR 模型典型调用链（使用 dnn_processing 滤镜 + OpenVINO 后端）

以下是用户执行命令时的完整调用链：

```bash
ffmpeg -i input.mp4 -vf \
  "format=yuv420p,dnn_processing=dnn_backend=openvino:model=sr_model.xml:input=input:output=output" \
  output.mp4
```

```
                          用户命令行
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│ 1. FFFilter ff_vf_dnn_processing                    │
│    .preinit = ff_dnn_filter_init_child_class        │  ← 初始化AVClass子类
│    .init    = init()                                │
│              └→ ff_dnn_init(&ctx->dnnctx,           │
│                             DFT_PROCESS_FRAME,      │
│                             context)                │
├─────────────────────────────────────────────────────┤
│ 2. dnn_filter_common.c :: ff_dnn_init()             │
│    ├── ff_get_dnn_module(DNN_OV) ─────────────────┐ │
│    │   返回 &ff_dnn_backend_openvino               │ │
│    └── dnn_module->load_model(ctx, func_type) ────┐│ │
│                                                   ││ │
├───────────────────────────────────────────────────┼┼─┤
│ 3. dnn_backend_openvino.c :: dnn_load_model_ov()  ││ │
│    ├── ov_core_create(&core) ◄════════════════════╡│ │  ← OpenVINO C API
│    ├── ov_core_read_model(core, model.xml) ◄══════╡│ │  ← OpenVINO C API
│    ├── model->get_input  = &get_input_ov          ││ │
│    ├── model->get_output = &get_output_ov         ││ │
│    └── return model ──────────────────────────────┘│ │
│                                                    │ │
├────────────────────────────────────────────────────┼─┤
│ 4. config_input() (输入配置阶段)                     │ │
│    └── ff_dnn_get_input(&ctx->dnnctx, &model_input)│ │
│        └→ model->get_input(model, input, name) ───┘ │
│           └→ get_input_ov()                          │
│              ├── ov_model_const_input_by_name()  ◄═══╡  ← OpenVINO C API
│              ├── ov_port_get_element_type()       ◄═══╡  ← OpenVINO C API
│              ├── ov_const_port_get_shape()        ◄═══╡  ← OpenVINO C API
│              └── return {dims, layout, dt}            │
│                                                      │
├──────────────────────────────────────────────────────┤
│ 5. config_output() (输出配置阶段)                     │
│    └── ff_dnn_get_output(ctx, inW, inH, &outW, &outH)│
│        └→ model->get_output() → get_output_ov()      │
│           ├── reshape (if input_resizable)             │
│           ├── init_model_ov() (首次编译)               │
│           │   ├── 创建预处理器                         │
│           │   ├── 设置输入布局NHWC, 数据类型U8         │
│           │   ├── 设置scale/mean预处理步骤             │
│           │   ├── ov_preprocess_build()  ◄════════════╡  ← OpenVINO C API
│           │   ├── ov_core_compile_model() ◄═══════════╡  ← OpenVINO C API
│           │   └── 创建推理请求和队列                    │
│           ├── execute_model_ov(试运行)                  │
│           └── return {output_width, output_height}     │
│                                                       │
├───────────────────────────────────────────────────────┤
│ 6. activate() (每帧处理)                               │
│    ├── ff_inlink_consume_frame() → in_frame            │
│    ├── ff_get_video_buffer() → out_frame               │
│    ├── ff_dnn_execute_model(ctx, in, out) ────────────┐│
│    │                                                  ││
│    │   dnn_filter_common.c :: ff_dnn_execute_model()  ││
│    │   └── dnn_module->execute_model(model, params) ──┘│
│    │                                                   │
│    │   dnn_backend_openvino.c :: dnn_execute_model_ov()│
│    │   ├── ff_dnn_fill_task(task, params)               │
│    │   ├── ff_queue_push_back(task_queue, task)         │
│    │   ├── extract_lltask_from_task()                   │
│    │   ├── ff_safe_queue_pop_front(request_queue)       │
│    │   └── execute_model_ov(request, lltask_queue) ───┐│
│    │                                                  ││
│    │   execute_model_ov():                            ││
│    │   ├── fill_model_input_ov():                     ││
│    │   │   ├── ov_tensor_create()         ◄═══════════╡│ ← OpenVINO C API
│    │   │   ├── ov_tensor_data() → data ptr            ││
│    │   │   ├── ov_infer_request_set_input_tensor() ◄══╡│ ← OpenVINO C API
│    │   │   └── ff_proc_from_frame_to_dnn(frame, input)││  ← 数据转换
│    │   │       └── sws_scale() (像素格式转换+归一化)   ││
│    │   │                                              ││
│    │   ├── [异步] ov_infer_request_set_callback() ◄═══╡│ ← OpenVINO C API
│    │   ├── [异步] ov_infer_request_start_async()  ◄═══╡│ ← OpenVINO C API
│    │   │                                              ││
│    │   └── [同步] ov_infer_request_infer()        ◄═══╡│ ← OpenVINO C API
│    │              └── infer_completion_callback() ─────┘│
│    │                                                   │
│    │   infer_completion_callback():                     │
│    │   ├── ov_infer_request_get_tensor_by_const_port()  │ ← OpenVINO C API
│    │   ├── ov_tensor_data() → output data               │ ← OpenVINO C API
│    │   ├── ff_proc_from_dnn_to_frame(out_frame, output) │ ← 数据转换
│    │   │   └── sws_scale() (反归一化+像素格式转换)       │
│    │   ├── task->inference_done++                       │
│    │   └── ff_safe_queue_push_back(requestq, request)   │
│    │                                                   │
│    ├── ff_dnn_get_result(ctx, &in, &out)               │
│    │   └── ff_dnn_get_result_common(task_queue, &in, &out)│
│    ├── copy_uv_planes() (YUV格式时)                     │
│    └── ff_filter_frame(outlink, out_frame)              │
│                                                        │
├────────────────────────────────────────────────────────┤
│ 7. uninit()                                            │
│    └── ff_dnn_uninit(&ctx->dnnctx)                     │
│        └── dnn_module->free_model(&model)              │
│            └── dnn_free_model_ov()                     │
│                ├── 释放推理请求队列                      │
│                ├── ov_compiled_model_free()  ◄═════════╡  ← OpenVINO C API
│                ├── ov_model_free()           ◄═════════╡  ← OpenVINO C API
│                └── ov_core_free()            ◄═════════╡  ← OpenVINO C API
└────────────────────────────────────────────────────────┘
```

### 7.2 调用层次总结

```
Layer 1: vf_dnn_processing.c (用户接口)
    │ 调用
    ▼
Layer 2: dnn_filter_common.c (封装层)
    │ ff_dnn_init() / ff_dnn_execute_model() / ff_dnn_get_result()
    │ 通过函数指针调用
    ▼
Layer 3: dnn_interface.c (路由层)
    │ ff_get_dnn_module() → 返回 &ff_dnn_backend_openvino
    │ 通过 DNNModule 函数指针调用
    ▼
Layer 4: dnn_backend_openvino.c (后端实现)
    │ dnn_load_model_ov() / dnn_execute_model_ov() / ...
    │ 调用 OpenVINO C API
    ▼
Layer 5: openvino/c/openvino.h (外部库)
    │ ov_core_create() / ov_core_read_model() / ov_infer_request_infer() / ...
    ▼
Layer 6: OpenVINO Runtime (C++ 实现)
    │ 实际硬件推理
    ▼
Hardware: CPU / GPU / VPU / ...
```

---

## 8. 数据流与 I/O 处理

### 8.1 数据类型

```c
typedef struct DNNData {
    void *data;           // 数据指针
    int dims[4];          // 维度 [N, C/H, H/W, W/C]
    DNNDataType dt;       // 数据类型 (DNN_FLOAT / DNN_UINT8)
    DNNColorOrder order;  // 颜色顺序 (DCO_BGR / DCO_RGB)
    DNNLayout layout;     // 布局 (DL_NCHW / DL_NHWC)
    float scale;          // 缩放因子
    float mean;           // 均值偏移
} DNNData;
```

### 8.2 帧处理数据流（DFT_PROCESS_FRAME）

```
AVFrame (YUV420P/RGB24/BGR24/GRAY8/...)
    │
    │ ff_proc_from_frame_to_dnn() 或 自定义 frame_pre_proc
    │ ├── 提取 Y 通道（YUV 格式）或全通道（RGB/BGR）
    │ ├── sws_scale() 像素格式转换
    │ ├── 归一化：pixel_value / scale - mean
    │ └── 布局转换：packed → planar (如需要)
    ▼
DNNData (float32, [1, C, H, W] 或 [1, H, W, C])
    │
    │ OpenVINO 推理
    ▼
DNNData (float32/uint8, [1, C, H', W'] 或 [1, H', W', C])
    │
    │ ff_proc_from_dnn_to_frame() 或 自定义 frame_post_proc
    │ ├── 反归一化：value * scale + mean
    │ ├── sws_scale() 像素格式转换
    │ └── 布局转换：planar → packed (如需要)
    ▼
AVFrame (输出格式，可能尺寸变化: H'×W')
```

### 8.3 预处理步骤详解

OpenVINO 后端的预处理可以通过两种方式实现：

**方式一：OpenVINO 内置预处理（推荐）**

在 `init_model_ov()` 中通过 OpenVINO PreProcessing API 配置：

```c
// 转换数据类型为 F32
ov_preprocess_preprocess_steps_convert_element_type(input_process_steps, F32);
// 减去均值
ov_preprocess_preprocess_steps_mean(input_process_steps, ctx->ov_option.mean);
// 除以 scale
ov_preprocess_preprocess_steps_scale(input_process_steps, ctx->ov_option.scale);
```

此方式利用 OpenVINO 的硬件加速预处理能力，效率更高。

**方式二：FFmpeg 软件预处理**

在 `dnn_io_proc.c` 中通过 `libswscale` 实现：

```c
// 帧到 DNN 数据：ff_proc_from_frame_to_dnn()
// DNN 数据到帧：ff_proc_from_dnn_to_frame()
```

### 8.4 SR 模型的 UV 通道处理

对于 YUV 格式，DNN 只处理 Y（亮度）通道，UV（色度）通道通过 `libswscale` 单独缩放：

```c
// vf_dnn_processing.c :: prepare_uv_scale()
if (isPlanarYUV(fmt)) {
    if (inlink->w != outlink->w || inlink->h != outlink->h) {
        // 输入输出尺寸不同（SR场景），创建UV缩放上下文
        ctx->sws_uv_scale = sws_getContext(
            src_w, src_h, AV_PIX_FMT_GRAY8,
            dst_w, dst_h, AV_PIX_FMT_GRAY8,
            SWS_BICUBIC, NULL, NULL, NULL);
    }
}

// activate() 中，推理完成后：
if (isPlanarYUV(in_frame->format))
    copy_uv_planes(ctx, out_frame, in_frame);
```

### 8.5 I/O 处理中的 Image Copy 详解

在整个 SR 推理路径中，图像数据经历了多次复制操作。以下是所有 image copy 的完整清单：

#### 8.5.1 dnn_io_proc.c 中的数据复制

**`ff_proc_from_frame_to_dnn()`（输入帧 → DNN 数据）：**

| 行号 | 操作 | 复制内容 | 触发条件 |
|------|------|---------|---------|
| L251-255 | `sws_scale()` | RGB24/BGR24 → GBRP 平面数据 | NCHW 布局的 RGB 输入 |
| L273-276 | `sws_scale()` | packed → float/UINT8 格式转换 | RGB24/BGR24 输入 |
| L280-282 | `av_image_copy_plane()` | GRAYF32 帧数据直接复制 | GRAYF32 格式直通 |
| L306-309 | `sws_scale()` | Y 通道 GRAY8 → float 格式 | YUV/GRAY8/NV12 输入 |

**`ff_proc_from_dnn_to_frame()`（DNN 数据 → 输出帧）：**

| 行号 | 操作 | 复制内容 | 触发条件 |
|------|------|---------|---------|
| L101-103 | `sws_scale()` | 输出数据格式转换（RGB） | RGB24/BGR24 输出 |
| L131-135 | `sws_scale()` | GBRP 平面 → packed RGB24/BGR24 | NCHW 布局的 RGB 输出 |
| L140-142 | `av_image_copy_plane()` | GRAYF32 数据直接复制 | GRAYF32 格式直通 |
| L166-168 | `sws_scale()` | GRAYF32 → GRAY8 转换 | YUV/GRAY8 输出 |

**`ff_frame_to_dnn_detect()`（检测预处理）：**

| 行号 | 操作 | 复制内容 |
|------|------|---------|
| L466-467 | `sws_scale()` | 全帧缩放到模型输入尺寸 + 格式转换 |

**`ff_frame_to_dnn_classify()`（分类预处理）：**

| 行号 | 操作 | 复制内容 |
|------|------|---------|
| L414-416 | `sws_scale()` | Bounding box 区域裁剪 + 缩放到模型输入尺寸 |

#### 8.5.2 vf_dnn_processing.c 中的数据复制

**`activate()`（主处理循环）：**

| 行号 | 操作 | 复制内容 |
|------|------|---------|
| L305 | `av_frame_copy_props()` | 帧元数据（pts、时间基等，非像素数据） |

**`copy_uv_planes()`（UV 通道复制/缩放）：**

| 行号 | 操作 | 复制内容 | 触发条件 |
|------|------|---------|---------|
| L233-235 | `av_image_copy_plane()` ×2 | U、V 平面直接复制 | 输入输出尺寸相同 |
| L238-239 | `sws_scale()` | NV12 UV 交织平面缩放 | NV12 + 尺寸变化（SR 场景） |
| L241-244 | `sws_scale()` ×2 | U、V 平面分别缩放 | YUV420P 等 + 尺寸变化 |

#### 8.5.3 dnn_backend_openvino.c 中的数据复制

**`fill_model_input_ov()`（填充输入张量）：**

| 行号 | 操作 | 说明 |
|------|------|------|
| L291 | `ov_tensor_data()` | 获取张量数据指针（**内存映射，非复制**） |
| L306-309 | 调用 `ff_proc_from_frame_to_dnn()` | 间接触发 dnn_io_proc 中的复制 |

**`infer_completion_callback()`（推理完成回调）：**

| 行号 | 操作 | 说明 |
|------|------|------|
| L366 | `ov_tensor_data()` | 获取输出张量指针（**内存映射，非复制**） |
| L450 | 调用 `ff_proc_from_dnn_to_frame()` | 间接触发 dnn_io_proc 中的复制 |

> ⚠ 注意：`ov_tensor_data()` 返回的是 OpenVINO 内部内存的指针，是零拷贝的内存映射操作。实际的数据复制发生在 `dnn_io_proc.c` 的 `sws_scale()` 和 `av_image_copy_plane()` 调用中。

#### 8.5.4 典型 SR 推理路径的完整 Copy 链

以 YUV420P 输入、4x SR 模型为例：

```
输入帧 (AVFrame, YUV420P, 640×480)
  │
  ├── [Copy 1] av_frame_copy_props(): 复制帧属性到输出帧
  │
  ├── [Copy 2] sws_scale() in ff_proc_from_frame_to_dnn():
  │     Y 通道 GRAY8 → GRAYF32（格式转换 + 归一化）
  │     ov_tensor_data() 返回目标指针（零拷贝映射）
  │
  │   ═══ OpenVINO 推理引擎执行 ═══
  │
  ├── [Copy 3] sws_scale() in ff_proc_from_dnn_to_frame():
  │     GRAYF32 → GRAY8（反归一化 + 格式转换）
  │     ov_tensor_data() 返回源指针（零拷贝映射）
  │
  ├── [Copy 4] sws_scale() in copy_uv_planes():
  │     U 通道缩放 (320×240 → 1280×960)
  │
  └── [Copy 5] sws_scale() in copy_uv_planes():
        V 通道缩放 (320×240 → 1280×960)

输出帧 (AVFrame, YUV420P, 2560×1920)
```

**总计：5 次数据复制操作**（1 次属性复制 + 2 次格式转换 + 2 次 UV 缩放）

对于 RGB24 输入 + NCHW 模型布局，复制次数更多（需要额外的 packed↔planar 转换）：
- 输入路径：packed RGB → planar GBR（1次） → 归一化（1次）= 2 次
- 输出路径：反归一化（1次） → planar GBR → packed RGB（1次）= 2 次
- 总计约 **5-6 次数据复制操作**

---

## 9. 异步执行机制

### 9.1 架构概览

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│ TaskItem     │     │ LLTaskItem  │     │ OVRequestItem    │
│ (高级任务)    │ 1:N │ (底层任务)   │ N:1 │ (推理请求)        │
│              │────▶│             │────▶│                  │
│ in_frame     │     │ task        │     │ infer_request    │
│ out_frame    │     │ bbox_index  │     │ lltasks[]        │
│ inference_   │     │             │     │ callback         │
│   todo/done  │     │             │     │                  │
└─────────────┘     └─────────────┘     └──────────────────┘
       ↑                   ↑                    ↑
   task_queue          lltask_queue        request_queue
   (Queue)             (Queue)             (SafeQueue)
```

### 9.2 队列角色

| 队列 | 类型 | 作用 |
|------|------|------|
| `task_queue` | Queue（非线程安全） | 存储所有提交的推理任务 |
| `lltask_queue` | Queue（非线程安全） | 存储待执行的底层任务 |
| `request_queue` | SafeQueue（线程安全） | 推理请求池（producer-consumer 模式） |

### 9.3 异步推理流程

```
1. dnn_execute_model_ov():
   ├── 创建 TaskItem，推入 task_queue
   ├── 提取 LLTaskItem，推入 lltask_queue
   ├── 从 request_queue 弹出 OVRequestItem
   └── execute_model_ov():
       ├── fill_model_input_ov() — 填充输入数据
       ├── ov_infer_request_set_callback() — 设置完成回调
       └── ov_infer_request_start_async() — 开始异步推理
                                              │
                                              │ (异步完成)
                                              ▼
2. infer_completion_callback():              (OpenVINO 线程)
   ├── 获取输出数据
   ├── 执行后处理
   ├── task->inference_done++
   └── ff_safe_queue_push_back(request_queue, request) — 回收请求

3. ff_dnn_get_result():                      (主线程)
   └── ff_dnn_get_result_common(task_queue):
       ├── peek task_queue 队首
       ├── if inference_done == inference_todo → DAST_SUCCESS
       ├── else → DAST_NOT_READY
       └── pop 并返回 in_frame/out_frame
```

### 9.4 并发请求数

```c
// 默认值计算
if (ctx->nireq <= 0) {
    ctx->nireq = av_cpu_count() / 2 + 1;
}
```

---

## 10. 大尺寸图片分块 SR 处理分析

### 10.1 当前状态：不支持分块处理

经过对 FFmpeg DNN 模块源码的全面检索，**FFmpeg 当前不支持分块（tile-based）超分辨率处理**。具体证据如下：

- 在所有 DNN 相关文件中（`libavfilter/dnn/*.c`、`libavfilter/vf_dnn_*.c`），未找到任何与 "tile"、"block"、"patch"、"crop"（分块裁剪）、"overlap"（重叠）、"split"（分割）、"chunk" 相关的逻辑
- `DFT_PROCESS_FRAME` 功能类型的定义注释为 "process the whole frame"（处理整帧）
- `vf_dnn_processing.c` 的 `activate()` 函数对每帧调用一次 `ff_dnn_execute_model()`，不做任何分块

### 10.2 全帧处理的代码证据

```c
// vf_dnn_processing.c :: activate() — 每帧完整处理，无分块
do {
    ret = ff_inlink_consume_frame(inlink, &in);
    if (ret > 0) {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        av_frame_copy_props(out, in);
        // 整帧送入模型推理，无分块逻辑
        if (ff_dnn_execute_model(&ctx->dnnctx, in, out) != 0) {
            return AVERROR(EIO);
        }
    }
} while (ret > 0);
```

### 10.3 相关选项澄清

**`input_resizable` 不是分块功能：**

该选项允许模型接受可变尺寸的输入，但它的实现是重塑（reshape）模型输入维度，而非将图片分块：

```c
// dnn_backend_openvino.c :: get_input_ov()
if (input_resizable) {
    input->dims[dnn_get_width_idx_by_layout(input->layout)] = -1;
    input->dims[dnn_get_height_idx_by_layout(input->layout)] = -1;
}

// dnn_backend_openvino.c :: get_output_ov() — reshape 模型输入
status = ov_model_reshape_single_input(ov_model->ov_model, partial_shape);
```

**`batch_size` 不是分块功能：**

OpenVINO 2.0 后端已明确不支持 batch_size > 1：

```c
// dnn_backend_openvino.c :: init_model_ov()
if (ctx->ov_option.batch_size > 1) {
    avpriv_report_missing_feature(ctx, "Do not support batch_size > 1 for now,"
                                       "change batch_size to 1.\n");
    ctx->ov_option.batch_size = 1;
}
```

### 10.4 大尺寸图片处理的限制

对于大尺寸图片的 SR 处理，当前存在以下限制：

1. **内存限制**：整帧加载到模型中，对于 4K/8K 图片，内存需求巨大（例如 4K RGB 输入 ≈ 24MB，4x SR 输出 ≈ 384MB，加上模型中间层可能需要数 GB）
2. **模型限制**：某些模型（如 SwinIR 的 Transformer 架构）对大尺寸输入的计算复杂度呈二次增长
3. **无边界重叠**：缺少分块之间的重叠和融合机制，即使外部分块也可能出现块边界伪影

### 10.5 变通方案

虽然 FFmpeg 内部不支持分块，但可以通过以下方式处理大尺寸图片：

**方案 A：FFmpeg 滤镜链外部分块**

```bash
# 1. 将大图切成小块（使用 crop 滤镜）
ffmpeg -i large_image.png -vf "crop=256:256:0:0" tile_0_0.png
ffmpeg -i large_image.png -vf "crop=256:256:256:0" tile_1_0.png
# ... 对每个块执行

# 2. 对每个块执行 SR
ffmpeg -i tile_0_0.png -vf \
  "dnn_processing=dnn_backend=openvino:model=sr.xml:input=x:output=y" \
  tile_0_0_sr.png

# 3. 使用外部工具拼接（FFmpeg 内无自动拼接+融合功能）
```

> ⚠ 此方案的主要问题是块边界处可能出现伪影，需要在外部实现重叠-裁剪（overlap-crop）策略。

**方案 B：使用 `input_resizable` 直接处理**

如果模型支持动态输入且内存足够，可以直接处理大图：

```bash
ffmpeg -i large_image.png -vf \
  "dnn_processing=dnn_backend=openvino:\
   model=sr.xml:input=x:output=y:\
   input_resizable=1" \
  output_sr.png
```

**方案 C：外部 Python 脚本分块处理**

推荐对于需要分块的场景，使用 Python + OpenVINO 直接实现分块+重叠+融合：

```python
import numpy as np
import openvino as ov

def tile_sr(image, model_path, tile_size=256, overlap=16, scale=4):
    core = ov.Core()
    model = core.compile_model(model_path, "CPU")
    h, w = image.shape[:2]
    output = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # 提取带重叠的块
            tile = image[y:y+tile_size, x:x+tile_size]
            # 推理
            result = model(tile)[0]
            # 融合到输出（裁剪重叠区域）
            # ... 融合逻辑
    return output
```

### 10.6 潜在的未来改进方向

如果要在 FFmpeg 中实现原生分块 SR 支持，需要：

1. 在 `vf_dnn_processing.c` 的 `activate()` 函数中添加分块逻辑
2. 实现重叠裁剪策略（overlap-crop），避免块边界伪影
3. 添加结果拼接和融合功能
4. 新增滤镜选项：`tile_size`、`tile_overlap`
5. 处理 YUV 格式下色度子采样对齐问题

---

## 11. 新图像 SR 模型集成步骤

### 11.1 概述

在 FFmpeg 中集成一个新的图像超分辨率（SR）模型主要涉及以下工作：

1. **模型准备**：将模型转换为 OpenVINO IR 格式
2. **确定模型接口**：输入/输出节点名、数据类型、布局
3. **配置预处理参数**：scale、mean、layout
4. **测试验证**：使用 `dnn_processing` 滤镜测试

对于大多数标准 SR 模型（单图像输入、单图像输出），**无需修改 FFmpeg 源代码**，只需使用现有的 `dnn_processing` 滤镜即可。

### 11.2 详细步骤

#### 步骤 1：模型准备与转换

**1a. 模型训练/获取**

获取或训练 SR 模型，常见框架：
- PyTorch（EDSR, RCAN, SwinIR, Real-ESRGAN 等）
- TensorFlow（SRCNN, ESPCN 等）
- ONNX 格式

**1b. 导出为 ONNX（如果使用 PyTorch）**

```python
import torch
model = load_your_sr_model()
model.eval()

# 定义示例输入（batch=1, channels=3, height=480, width=640）
dummy_input = torch.randn(1, 3, 480, 640)

# 导出 ONNX
torch.onnx.export(
    model, dummy_input, "sr_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "output": {2: "out_height", 3: "out_width"}
    },
    opset_version=11
)
```

**1c. 转换为 OpenVINO IR 格式**

```bash
# 安装 OpenVINO 开发工具
pip install openvino-dev

# 使用 Model Optimizer 转换
mo --input_model sr_model.onnx \
   --output_dir ./ir_model/ \
   --input_shape "[1,3,480,640]" \
   --data_type FP32

# 生成文件：
# ir_model/sr_model.xml  — 模型结构
# ir_model/sr_model.bin  — 模型权重
```

或使用 OpenVINO Python API 直接转换：

```python
import openvino as ov
core = ov.Core()
model = core.read_model("sr_model.onnx")
ov.serialize(model, "sr_model.xml")
```

#### 步骤 2：确定模型接口参数

**关键参数清单：**

| 参数 | 说明 | 查看方法 |
|------|------|---------|
| `input_name` | 输入节点名称 | `ov.Core().read_model("model.xml").inputs` |
| `output_name` | 输出节点名称 | `ov.Core().read_model("model.xml").outputs` |
| `input_layout` | 输入布局 | 通常 NCHW（PyTorch 模型）或 NHWC |
| `input_dtype` | 输入数据类型 | 通常 FP32 |
| `scale` | 输入缩放因子 | 取决于模型训练时的预处理 |
| `mean` | 输入均值偏移 | 取决于模型训练时的预处理 |
| `scale_factor` | SR 放大倍数 | 模型设计参数（如 2x, 4x） |

**使用 Python 查看模型信息：**

```python
import openvino as ov
core = ov.Core()
model = core.read_model("sr_model.xml")

# 查看输入信息
for inp in model.inputs:
    print(f"Input: name={inp.any_name}, shape={inp.shape}, "
          f"element_type={inp.element_type}")

# 查看输出信息
for out in model.outputs:
    print(f"Output: name={out.any_name}, shape={out.shape}, "
          f"element_type={out.element_type}")
```

#### 步骤 3：确定预处理策略

根据模型训练时的预处理方式，选择合适的 `scale` 和 `mean` 参数：

| 模型类型 | 输入范围 | scale | mean | 说明 |
|----------|---------|-------|------|------|
| Y 通道 SR（如 SRCNN） | [0, 1] | 255 | 0 | Y 通道归一化到 [0,1] |
| RGB SR（如 EDSR） | [0, 255] | 1 | 0 | 直接使用原始像素值 |
| RGB SR（归一化） | [0, 1] | 255 | 0 | RGB 归一化到 [0,1] |
| RGB SR（ImageNet 归一化） | 特殊 | 自定义 | 自定义 | 需要自定义预/后处理 |

⚠ **注意**：FFmpeg OpenVINO 后端的 `scale` 选项会将输入**除以** scale 值（即 `value / scale`），默认对 `DFT_PROCESS_FRAME` 类型模型 scale=255。

#### 步骤 4：使用 dnn_processing 滤镜测试

**场景 A：Y 通道 SR 模型（如 SRCNN，输入 [0,1] 的 Y 通道浮点值）**

```bash
ffmpeg -i input.mp4 -vf \
  "format=yuv420p,\
   dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=input:\
   output=output:\
   nireq=4:\
   async=1" \
  -y output.mp4
```

此场景下：
- `format=yuv420p`：转换为 YUV420P，DNN 只处理 Y 通道
- scale 默认为 255（Y 通道像素 0-255 归一化到 0-1）
- UV 通道通过 bicubic 插值自动缩放到输出尺寸

**场景 B：RGB 3通道 SR 模型（输入 [0,1] 的 RGB 浮点值）**

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=input:\
   output=output:\
   layout=nhwc:\
   scale=255" \
  -y output.mp4
```

**场景 C：RGB 3通道 SR 模型（输入 [0,255] 的 UINT8 值）**

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=input:\
   output=output:\
   layout=nhwc:\
   scale=1:\
   mean=0" \
  -y output.mp4
```

**场景 D：支持动态输入尺寸的模型**

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=input:\
   output=output:\
   input_resizable=1" \
  -y output.mp4
```

#### 步骤 5：自定义预/后处理（如需要）

如果模型需要特殊的预/后处理（例如 ImageNet 标准化、特殊的通道排列等），需要修改 FFmpeg 源代码：

**5a. 创建新的滤镜文件（可选）**

如果 `dnn_processing` 不能满足需求，可以创建专用滤镜：

```c
// libavfilter/vf_dnn_sr_custom.c
#include "dnn_filter_common.h"

typedef struct CustomSRContext {
    const AVClass *class;
    DnnContext dnnctx;
    // ... 自定义参数
} CustomSRContext;

// 自定义预处理：AVFrame → DNNData
static int custom_pre_proc(AVFrame *frame, DNNData *model_input,
                           AVFilterContext *filter_ctx) {
    // 实现自定义预处理逻辑
    // 例如：ImageNet 标准化、特殊的通道排列等
    return 0;
}

// 自定义后处理：DNNData → AVFrame
static int custom_post_proc(AVFrame *frame, DNNData *model_output,
                            AVFilterContext *filter_ctx) {
    // 实现自定义后处理逻辑
    // 例如：反标准化、clip 到 [0,255] 等
    return 0;
}

static av_cold int init(AVFilterContext *context) {
    CustomSRContext *ctx = context->priv;
    int ret = ff_dnn_init(&ctx->dnnctx, DFT_PROCESS_FRAME, context);
    if (ret < 0) return ret;

    // 注册自定义预/后处理
    ff_dnn_set_frame_proc(&ctx->dnnctx, custom_pre_proc, custom_post_proc);
    return 0;
}
```

**5b. 注册新滤镜**

在 `libavfilter/allfilters.c` 中添加：

```c
extern const FFFilter ff_vf_dnn_sr_custom;
```

在 `libavfilter/Makefile` 中添加：

```makefile
OBJS-$(CONFIG_DNN_SR_CUSTOM_FILTER) += vf_dnn_sr_custom.o
```

#### 步骤 6：构建与编译

```bash
# 配置（启用 OpenVINO 支持）
./configure --enable-libopenvino

# 编译
make -j$(nproc)
```

#### 步骤 7：验证与调试

**7a. 查看模型输入/输出信息（详细日志）**

```bash
ffmpeg -v verbose -i input.mp4 -vf \
  "dnn_processing=dnn_backend=openvino:model=sr_model.xml:input=x:output=y" \
  -frames:v 1 -y output.png
```

**7b. 验证输出尺寸**

```bash
ffprobe output.mp4
# 检查视频流的 width 和 height 是否为预期的放大尺寸
```

**7c. 性能测试**

```bash
# 异步模式 + 多请求
ffmpeg -i input.mp4 -vf \
  "dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=x:output=y:\
   nireq=4:async=1:\
   device=CPU" \
  -benchmark -y output.mp4
```

### 11.3 高级集成场景

#### 10.3.1 使用 GPU 推理

```bash
ffmpeg -i input.mp4 -vf \
  "dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=x:output=y:\
   device=GPU" \
  -y output.mp4
```

#### 10.3.2 批处理推理

```bash
# OpenVINO 2.0 暂不支持 batch_size > 1
# 但可以通过 nireq 实现流水线并行
ffmpeg -i input.mp4 -vf \
  "dnn_processing=dnn_backend=openvino:\
   model=sr_model.xml:\
   input=x:output=y:\
   nireq=8:async=1" \
  -y output.mp4
```

#### 10.3.3 与其他滤镜组合

```bash
# SR + 去噪 + 锐化 管道
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:model=denoise.xml:input=x:output=y,\
   dnn_processing=dnn_backend=openvino:model=sr_4x.xml:input=x:output=y,\
   unsharp=5:5:1.0" \
  -y output.mp4
```

### 11.4 集成检查清单

| # | 检查项 | 说明 |
|---|--------|------|
| 1 | ✅ 模型格式 | 确认已转换为 OpenVINO IR (.xml + .bin) |
| 2 | ✅ 输入/输出名称 | 使用工具确认正确的节点名称 |
| 3 | ✅ 输入布局 | 确认 NCHW 或 NHWC，通过 `layout` 选项指定 |
| 4 | ✅ 预处理参数 | 确认 `scale` 和 `mean` 与训练时一致 |
| 5 | ✅ 像素格式 | 选择正确的 `format` 滤镜（rgb24/yuv420p 等） |
| 6 | ✅ 动态输入 | 如需动态输入尺寸，设置 `input_resizable=1` |
| 7 | ✅ 输出尺寸 | 通过 `config_output` 自动获取（无需手动指定） |
| 8 | ✅ UV 通道 | YUV 格式自动处理（bicubic 缩放） |
| 9 | ✅ 异步推理 | 建议启用 `async=1` 并设置合适的 `nireq` |
| 10 | ✅ 设备选择 | 根据硬件选择 CPU/GPU/VPU |

### 11.5 常见 SR 模型集成示例

#### SRCNN (Y 通道，TensorFlow 原始模型)

```bash
ffmpeg -i input.mp4 -vf \
  "format=yuv420p,\
   dnn_processing=dnn_backend=openvino:\
   model=srcnn.xml:\
   input=x:output=y" \
  -y output_srcnn.mp4
```

#### EDSR (RGB 3通道, 4x 放大)

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=edsr_4x.xml:\
   input=input:output=output:\
   layout=nchw:\
   scale=255" \
  -y output_edsr.mp4
```

#### Real-ESRGAN (RGB 3通道, 4x 放大)

> **关于 Real-ESRGAN 集成说明**：FFmpeg **没有** Real-ESRGAN 的专用实现代码。Real-ESRGAN 通过通用的 `dnn_processing` 滤镜加载，与其他所有 SR 模型（EDSR、SwinIR 等）使用完全相同的代码路径。用户只需将 Real-ESRGAN 模型转换为 OpenVINO IR 格式（.xml + .bin），然后通过以下命令使用：

**模型准备步骤**：

```bash
# 1. 获取 Real-ESRGAN 的 PyTorch 模型
# 2. 导出为 ONNX
python -c "
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(torch.load('RealESRGAN_x4plus.pth')['params_ema'])
model.eval()
torch.onnx.export(model, torch.randn(1,3,64,64), 'realesrgan_x4.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'h',3:'w'}})
"
# 3. 转换为 OpenVINO IR
mo --input_model realesrgan_x4.onnx --output_dir ./
```

**FFmpeg 使用命令**：

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=realesrgan_4x.xml:\
   input=input:output=output:\
   layout=nchw:\
   scale=255:\
   input_resizable=1:\
   async=1:\
   nireq=2" \
  -y output_realesrgan.mp4
```

> ⚠ **注意**：Real-ESRGAN 模型较大（RRDBNet 约 64MB），对于大尺寸输入帧，内存消耗可能很高。如遇内存不足问题，考虑降低输入分辨率或参考第 10 节的变通方案。

#### SwinIR (RGB 3通道, 轻量级 SR)

```bash
ffmpeg -i input.mp4 -vf \
  "format=rgb24,\
   dnn_processing=dnn_backend=openvino:\
   model=swinir_lightweight_x4.xml:\
   input=input:output=output:\
   layout=nchw:\
   scale=255:\
   device=GPU" \
  -y output_swinir.mp4
```

---

## 12. 附录：关键数据结构参考

### 12.1 数据类型枚举

```c
typedef enum {DNN_FLOAT = 1, DNN_UINT8 = 4} DNNDataType;

typedef enum {
    DCO_NONE,
    DCO_BGR,    // OpenVINO Model Zoo 默认输入格式
    DCO_RGB,
} DNNColorOrder;

typedef enum {
    DL_NONE,
    DL_NCHW,    // PyTorch 默认布局
    DL_NHWC,    // TensorFlow 默认布局
} DNNLayout;
```

### 12.2 异步状态

```c
typedef enum {
    DAST_FAIL,          // 错误
    DAST_EMPTY_QUEUE,   // 队列为空
    DAST_NOT_READY,     // 推理未完成
    DAST_SUCCESS        // 成功获取结果
} DNNAsyncStatusType;
```

### 12.3 任务结构

```c
typedef struct TaskItem {
    void *model;                // 后端模型指针
    AVFrame *in_frame;          // 输入帧
    AVFrame *out_frame;         // 输出帧
    const char *input_name;     // 输入节点名
    const char **output_names;  // 输出节点名数组
    uint8_t async;              // 是否异步
    uint8_t do_ioproc;          // 是否进行 I/O 处理
    uint32_t nb_output;         // 输出数量
    uint32_t inference_todo;    // 待完成推理数
    uint32_t inference_done;    // 已完成推理数
} TaskItem;

typedef struct LastLevelTaskItem {
    TaskItem *task;             // 关联的高级任务
    uint32_t bbox_index;        // Bounding box 索引（分类用）
} LastLevelTaskItem;
```

### 12.4 构建系统集成

```makefile
# libavfilter/dnn/Makefile
OBJS-$(CONFIG_DNN)                    += dnn/dnn_interface.o
OBJS-$(CONFIG_DNN)                    += dnn/dnn_io_proc.o
OBJS-$(CONFIG_DNN)                    += dnn/queue.o
OBJS-$(CONFIG_DNN)                    += dnn/safe_queue.o
OBJS-$(CONFIG_DNN)                    += dnn/dnn_backend_common.o

DNN-OBJS-$(CONFIG_LIBTENSORFLOW)      += dnn/dnn_backend_tf.o
DNN-OBJS-$(CONFIG_LIBOPENVINO)        += dnn/dnn_backend_openvino.o
DNN-OBJS-$(CONFIG_LIBTORCH)           += dnn/dnn_backend_torch.o

OBJS-$(CONFIG_DNN)                    += $(DNN-OBJS-yes)
```

编译配置：

```bash
./configure --enable-libopenvino  # 启用 OpenVINO 支持
./configure --enable-libtensorflow  # 启用 TensorFlow 支持
./configure --enable-libtorch  # 启用 LibTorch 支持
```

---

*本文档基于 FFmpeg 源码分析生成，涵盖了 libavfilter → DNN Interface → OpenVINO Backend → OpenVINO C Library 的完整调用链，以及新图像 SR 模型的集成方法。*
